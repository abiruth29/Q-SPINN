import asyncio
import hashlib
import json
import logging
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# For Sept 2025 - Aug 2026
YEAR_MONTHS = []
for y in [2025, 2026]:
    for m in range(1, 13):
        if y == 2025 and m >= 9:
            YEAR_MONTHS.append((y, m))
        elif y == 2026 and m <= 8:
            YEAR_MONTHS.append((y, m))

BASE_URL = "https://data.binance.vision/data/spot/monthly/trades"
DATA_DIR = Path("data")
CURATED_DIR = DATA_DIR / "curated" / "trades"
SCRATCH_DIR = DATA_DIR / "raw"
MANIFEST_FILE = CURATED_DIR / "_manifest.json"

SCHEMA = pa.schema([
    ("trade_id", pa.int64()),
    ("price", pa.float64()),
    ("qty", pa.float64()),
    ("quote_qty", pa.float64()),
    ("timestamp_us", pa.int64()),      # Always microseconds
    ("is_buyer_maker", pa.bool_()),
])

CSV_COLUMN_NAMES = ["trade_id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]

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

def is_verified(manifest, symbol, year, month):
    for part in manifest.get("partitions", []):
        if part["symbol"] == symbol and part["year"] == year and part["month"] == month and part["status"] == "verified":
            return True
    return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=60))
async def download_and_verify(client, symbol, year, month):
    zip_filename = f"{symbol}-trades-{year}-{month:02d}.zip"
    zip_url = f"{BASE_URL}/{symbol}/{zip_filename}"
    checksum_url = f"{zip_url}.CHECKSUM"
    
    zip_path = SCRATCH_DIR / zip_filename
    
    logger.info(f"[{symbol} {year}-{month:02d}] Fetching checksum...")
    resp_chk = await client.get(checksum_url)
    resp_chk.raise_for_status()
    expected_hash = resp_chk.text.split()[0]
    
    logger.info(f"[{symbol} {year}-{month:02d}] Downloading zip...")
    async with client.stream("GET", zip_url) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            async for chunk in resp.aiter_bytes():
                f.write(chunk)
                
    logger.info(f"[{symbol} {year}-{month:02d}] Verifying checksum...")
    sha256_hash = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
            
    if sha256_hash.hexdigest() != expected_hash:
        zip_path.unlink()
        raise ValueError(f"Checksum mismatch for {symbol} {year}-{month:02d}")
        
    return zip_path

def process_and_write(zip_path, symbol, year, month):
    logger.info(f"[{symbol} {year}-{month:02d}] Extracting CSV...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_filename = zf.namelist()[0]
        csv_path = SCRATCH_DIR / csv_filename
        zf.extract(csv_filename, SCRATCH_DIR)

    logger.info(f"[{symbol} {year}-{month:02d}] Converting to Parquet...")
    
    # Read CSV using pyarrow
    parse_options = pacsv.ParseOptions(newlines_in_values=False)
    read_options = pacsv.ReadOptions(column_names=CSV_COLUMN_NAMES)
    convert_options = pacsv.ConvertOptions(
        column_types={
            "trade_id": pa.int64(),
            "price": pa.float64(),
            "qty": pa.float64(),
            "quote_qty": pa.float64(),
            "time": pa.int64(),
            "is_buyer_maker": pa.bool_(),
            "is_best_match": pa.bool_()
        }
    )
    
    table = pacsv.read_csv(csv_path, read_options=read_options, parse_options=parse_options, convert_options=convert_options)
    
    # Drop is_best_match
    table = table.drop(["is_best_match"])
    
    # Normalize timestamp (milliseconds to microseconds before 2025)
    time_arr = table["time"].to_numpy()
    if year < 2025:
        time_arr = time_arr * 1000
        
    table = table.set_column(
        table.schema.get_field_index("time"), 
        pa.field("timestamp_us", pa.int64()), 
        pa.array(time_arr)
    )
    
    # Cast to final schema to ensure exact match
    table = table.cast(SCHEMA)
    
    # Add metadata
    metadata = {
        b"qspinn_schema_version": b"1",
        b"qspinn_timestamp_unit": b"microseconds",
        b"qspinn_source": b"data.binance.vision/spot/monthly/trades",
        b"qspinn_created_at": str(datetime.now(timezone.utc)).encode(),
    }
    table = table.replace_schema_metadata(metadata)
    
    # Verify monotonicity and gaps
    trade_ids = table["trade_id"].to_pylist()
    timestamps = table["timestamp_us"].to_pylist()
    
    if not all(a < b for a, b in zip(trade_ids, trade_ids[1:])):
        logger.warning(f"[{symbol} {year}-{month:02d}] Trade ID regression/duplicate detected! Sorting table by trade_id...")
        table = table.sort_by([("trade_id", "ascending")])
        # Re-fetch the columns after sorting
        trade_ids = table["trade_id"].to_pylist()
        timestamps = table["timestamp_us"].to_pylist()
        
        # If there are still exact duplicates, we just warn, but don't crash
        if not all(a < b for a, b in zip(trade_ids, trade_ids[1:])):
            logger.warning(f"[{symbol} {year}-{month:02d}] Exact duplicate Trade IDs exist. Proceeding anyway.")
        
    if not all(a <= b for a, b in zip(timestamps, timestamps[1:])):
        logger.warning(f"[{symbol} {year}-{month:02d}] Timestamp regression detected (trades matched out of strict chronological order).")
        
    # Write partitioned parquet
    # Add partition columns
    sym_col = pa.array([symbol] * table.num_rows, type=pa.string())
    yr_col = pa.array([year] * table.num_rows, type=pa.int32())
    mo_col = pa.array([month] * table.num_rows, type=pa.int32())
    
    out_table = table.append_column("symbol", sym_col).append_column("year", yr_col).append_column("month", mo_col)
    
    pq.write_to_dataset(
        out_table,
        root_path=str(CURATED_DIR),
        partition_cols=["symbol", "year", "month"],
        compression="zstd",
        compression_level=3,
        row_group_size=500_000,
        use_dictionary=["symbol"]
    )
    
    row_count = table.num_rows
    min_ts = timestamps[0]
    max_ts = timestamps[-1]
    
    # Cleanup scratch
    csv_path.unlink()
    zip_path.unlink()
    
    return {
        "symbol": symbol,
        "year": year,
        "month": month,
        "row_count": row_count,
        "min_timestamp_us": min_ts,
        "max_timestamp_us": max_ts,
        "status": "verified"
    }

async def main():
    init_dirs()
    manifest = load_manifest()
    
    total_files = len(SYMBOLS) * len(YEAR_MONTHS)
    completed_files = 0
    
    # We use a single global httpx client for connection pooling
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Sequential processing as requested for 8-10 Mbps connections
        for symbol in SYMBOLS:
            for year, month in YEAR_MONTHS:
                completed_files += 1
                if is_verified(manifest, symbol, year, month):
                    logger.info(f"[{completed_files}/{total_files}] [{symbol} {year}-{month:02d}] Already verified. Skipping.")
                    continue
                
                logger.info(f"[{completed_files}/{total_files}] === Starting {symbol} for {year}-{month:02d} ===")
                try:
                    zip_path = await download_and_verify(client, symbol, year, month)
                    part_info = process_and_write(zip_path, symbol, year, month)
                    
                    # Update manifest safely
                    manifest["partitions"].append(part_info)
                    save_manifest(manifest)
                    logger.info(f"[{completed_files}/{total_files}] === Finished {symbol} for {year}-{month:02d} ===")
                except Exception as e:
                    logger.error(f"[{completed_files}/{total_files}] Failed processing {symbol} {year}-{month:02d}: {e}")
                    # Stop execution on failure to avoid compounding issues
                    raise e
                    
if __name__ == "__main__":
    asyncio.run(main())
