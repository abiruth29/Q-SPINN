import asyncio
import hashlib
import json
import logging
import os
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import websockets
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup logging to both console and file
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('ingestor_live.log')
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("Q-SPINN-Ingestor")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BASE_URL_MONTHLY = "https://data.binance.vision/data/spot/monthly/trades"
BASE_URL_DAILY = "https://data.binance.vision/data/spot/daily/trades"
DATA_DIR = Path("data")
CURATED_DIR = DATA_DIR / "curated" / "trades"
SCRATCH_DIR = DATA_DIR / "raw"
MANIFEST_FILE = CURATED_DIR / "_manifest.json"

# Target: Sept 2025 to Current Date (October 2026)
START_YEAR = 2025
START_MONTH = 9

def init_dirs():
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "partitions": []}, f)

def load_manifest():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_manifest(manifest):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def is_verified(manifest, symbol, date_str):
    for part in manifest.get("partitions", []):
        if part.get("symbol") != symbol or part.get("status") != "verified":
            continue
            
        # Check new format
        if part.get("date") == date_str:
            return True
            
        # Check legacy format (from older script)
        if "year" in part and "month" in part:
            legacy_date = f"{part['year']}-{part['month']:02d}"
            if legacy_date == date_str:
                return True
                
    return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=60))
async def download_archive(client, url_base, symbol, filename):
    zip_url = f"{url_base}/{symbol}/{filename}"
    checksum_url = f"{zip_url}.CHECKSUM"
    zip_path = SCRATCH_DIR / filename
    
    logger.info(f"[{symbol} {filename}] Fetching checksum...")
    resp_chk = await client.get(checksum_url)
    if resp_chk.status_code == 404:
        logger.warning(f"Data not yet available for {filename}")
        return None
    resp_chk.raise_for_status()
    expected_hash = resp_chk.text.split()[0]
    
    logger.info(f"[{symbol} {filename}] Downloading zip...")
    async with client.stream("GET", zip_url) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            async for chunk in resp.aiter_bytes():
                f.write(chunk)
                
    logger.info(f"[{symbol} {filename}] Verifying checksum...")
    sha256_hash = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
            
    if sha256_hash.hexdigest() != expected_hash:
        zip_path.unlink()
        raise ValueError(f"Checksum mismatch for {filename}")
        
    return zip_path

def extract_and_convert(zip_path, symbol, date_str):
    logger.info(f"[{symbol} {date_str}] Extracting CSV...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_filename = z.namelist()[0]
        z.extract(csv_filename, SCRATCH_DIR)
        
    csv_path = SCRATCH_DIR / csv_filename
    parquet_dir = CURATED_DIR / f"symbol={symbol}"
    parquet_dir.mkdir(exist_ok=True)
    parquet_path = parquet_dir / f"{date_str}.parquet"
    
    logger.info(f"[{symbol} {date_str}] Converting to Parquet...")
    table = pacsv.read_csv(csv_path, read_options=pacsv.ReadOptions(column_names=["trade_id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]))
    
    # Check for Trade ID regression
    trade_ids = table.column("trade_id").to_numpy()
    if not (trade_ids[:-1] <= trade_ids[1:]).all():
        logger.warning(f"[{symbol} {date_str}] Trade ID regression! Sorting table...")
        table = table.sort_by([("trade_id", "ascending")])
        
    pq.write_table(table, parquet_path, compression="snappy")
    
    csv_path.unlink()
    zip_path.unlink()
    
    return str(parquet_path)

async def backfill_historical():
    init_dirs()
    manifest = load_manifest()
    
    now = datetime.now(timezone.utc)
    current_year, current_month = now.year, now.month
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Backfill Monthlies (up to last full month)
        for y in range(START_YEAR, current_year + 1):
            for m in range(1, 13):
                if y == START_YEAR and m < START_MONTH:
                    continue
                if y == current_year and m >= current_month: # Current month is not available as a monthly zip yet
                    continue
                
                date_str = f"{y}-{m:02d}"
                for symbol in SYMBOLS:
                    if is_verified(manifest, symbol, date_str):
                        logger.info(f"[{symbol} {date_str}] Already verified. Skipping.")
                        continue
                        
                    zip_name = f"{symbol}-trades-{date_str}.zip"
                    zip_path = await download_archive(client, BASE_URL_MONTHLY, symbol, zip_name)
                    if zip_path:
                        pq_path = extract_and_convert(zip_path, symbol, date_str)
                        manifest["partitions"].append({"symbol": symbol, "date": date_str, "status": "verified", "path": pq_path})
                        save_manifest(manifest)

        # 2. Backfill Dailies (for the current month up to yesterday)
        first_of_month = datetime(current_year, current_month, 1, tzinfo=timezone.utc)
        yesterday = now - timedelta(days=1)
        
        current_day = first_of_month
        while current_day <= yesterday:
            date_str = current_day.strftime("%Y-%m-%d")
            for symbol in SYMBOLS:
                if is_verified(manifest, symbol, date_str):
                    logger.info(f"[{symbol} {date_str}] Already verified (Daily). Skipping.")
                    continue
                    
                zip_name = f"{symbol}-trades-{date_str}.zip"
                zip_path = await download_archive(client, BASE_URL_DAILY, symbol, zip_name)
                if zip_path:
                    pq_path = extract_and_convert(zip_path, symbol, date_str)
                    manifest["partitions"].append({"symbol": symbol, "date": date_str, "status": "verified", "path": pq_path})
                    save_manifest(manifest)
            
            current_day += timedelta(days=1)

async def fetch_rest_snapshot(client: httpx.AsyncClient, symbol: str, limit: int = 1000):
    """
    Fetches a deep order book snapshot via Binance REST API.
    Required to initialize the local order book before applying WebSocket deltas.
    """
    url = f"https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol.upper(), "limit": limit}
    try:
        logger.info(f"[{symbol}] Fetching REST Order Book Snapshot (limit={limit})...")
        response = await client.get(url, params=params, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        logger.info(f"[{symbol}] Snapshot acquired: {len(data['bids'])} bids, {len(data['asks'])} asks. (Update ID: {data['lastUpdateId']})")
        return data
    except Exception as e:
        logger.error(f"[{symbol}] Failed to fetch REST snapshot: {e}")
        return None

async def stream_live(callback=None):
    streams = []
    for s in SYMBOLS:
        streams.append(f"{s.lower()}@depth20@100ms")
        streams.append(f"{s.lower()}@trade")
        
    stream_url = f"wss://stream.binance.com/stream?streams={'/'.join(streams)}"
    logger.info(f"Connecting to Binance Live WebSocket: {stream_url}")
    
    # Snapshot Sync: Fetch deep REST snapshots to initialize the local order book state
    # before we start processing live deltas.
    async with httpx.AsyncClient() as client:
        snapshots = {}
        for s in SYMBOLS:
            snapshots[s] = await fetch_rest_snapshot(client, s, limit=1000)
    
    while True:
        try:
            async with websockets.connect(stream_url) as websocket:
                logger.info("Successfully connected to live stream!")
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    stream_name = data.get('stream', 'unknown')
                    payload = data.get('data', {})
                    
                    if callback:
                        await callback(stream_name, payload)
                    else:
                        if 'trade' in stream_name:
                            pass
                        elif 'depth' in stream_name:
                            pass
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            logger.error(f"WebSocket error: {e}. Traceback:\n{traceback_str}\nReconnecting in 5 seconds...")
            await asyncio.sleep(5)

async def main():
    logger.info("=== Q-SPINN Data Ingestor Started ===")
    
    logger.info("1. Bootstrapping Historical Backfill (Monthly + Daily)...")
    await backfill_historical()
    logger.info("Backfill complete and up-to-date.")
    
    logger.info("2. Launching Live Streaming WebSocket...")
    await stream_live()

if __name__ == "__main__":
    asyncio.run(main())
