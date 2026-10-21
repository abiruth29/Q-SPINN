import sys
import time
import json
import asyncio
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments'))

from ingestor import stream_live
from exp_B_hybrid_fi2010.train import QSPINN, N_ASSETS

from lob_extractor import FI2010Extractor
from live_contract import make_contract, load_contract, contracts_match, LiveWindow
from exp_B_hybrid_fi2010.train import SEQ_LEN, N_FEATURES

# Initialize the Q-SPINN.  Order matters (review item 10): decide on the input/protocol contract BEFORE loading
# any tensors, and never load legacy weights into the current architecture.
device = 'cpu'     # forced to CPU for live inference to prevent background CUDA context crashes
ckpt_path = ROOT / "experiments" / "exp_B_hybrid_fi2010" / "qspinn_fi2010.pt"
LIVE_CONTRACT = make_contract(SEQ_LEN, N_FEATURES, "binance_depth_extractor", "online_zscore", "crypto")
TRAIN_CONTRACT = load_contract(ckpt_path.with_name("qspinn_fi2010_contract.json"))
CONTRACT_OK, CONTRACT_MISMATCH = contracts_match(TRAIN_CONTRACT, LIVE_CONTRACT)
MODE = "PRODUCTION" if CONTRACT_OK else "DEMONSTRATION (weights not trained on this input contract)"
WEIGHTS_ERROR = None

model = QSPINN().to(device)
if not ckpt_path.exists():
    WEIGHTS_ERROR = (f"no weights at {ckpt_path.name}. Legacy v3.0 weights are archived in experiments/legacy_v3.0/ "
                     "and are NOT loadable (different generator/protocol). Train with v3.1 first.")
elif TRAIN_CONTRACT is None or TRAIN_CONTRACT.get("protocol") != LIVE_CONTRACT["protocol"]:
    WEIGHTS_ERROR = (f"weights have no matching contract/protocol (found {None if TRAIN_CONTRACT is None else TRAIN_CONTRACT.get('protocol')!r}, "
                     f"need {LIVE_CONTRACT['protocol']!r}). Refusing to load them.")
else:
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("✅ Loaded protocol-matched weights")
    except Exception as e:          # shape/key mismatch = wrong architecture version
        WEIGHTS_ERROR = f"weights do not fit the current architecture: {type(e).__name__}: {str(e)[:200]}"
model.eval()
window = LiveWindow(SEQ_LEN, N_FEATURES)

LOG_FILE = ROOT / "live_inference.jsonl"
last_write = time.time()

# Store last seen prices and order books
prices = {"btc": 64000.0, "eth": 3500.0, "sol": 150.0}
extractors = {"btc": FI2010Extractor(), "eth": FI2010Extractor(), "sol": FI2010Extractor()}
latest_features = {"btc": np.zeros(144), "eth": np.zeros(144), "sol": np.zeros(144)}

from collections import deque
omega_history = deque(maxlen=600)  # 5 minutes at 2Hz (dt=0.5s)

async def process_tick(stream_name, payload):
    global last_write
    
    symbol = ""
    if 'btc' in stream_name: symbol = "btc"
    elif 'eth' in stream_name: symbol = "eth"
    elif 'sol' in stream_name: symbol = "sol"
    
    if not symbol: return
    
    # Extract real prices from the live trade stream
    if 'trade' in stream_name:
        p = float(payload.get('p', 0))
        prices[symbol] = p
        
    # Reconstruct FI-2010 144 features from Depth stream
    elif 'depth' in stream_name:
        bids = payload.get('b', [])
        asks = payload.get('a', [])
        feats = extractors[symbol].update_and_extract(bids, asks)
        if feats is not None:
            latest_features[symbol] = feats
        
    # Throttle inference and logging to 2 Hz so we don't overwhelm the dashboard file
    if time.time() - last_write < 0.5:
        return
    last_write = time.time()
    
    # Task 17: fixed-length window identical to training (no duplicate padding); wait until it is full
    window.push(latest_features["btc"])
    if not window.ready:
        return
    features = window.tensor().to(device)

    # LIVE INFERENCE: heavy path -> generator S; omega = lambda_max(S) is a LATENT-FLOW DIAGNOSTIC,
    # not a validated crash probability (see docs/RUNBOOK_TEAMMATE.md, item 10).
    with torch.no_grad():
        heavy, L_pde, H, S, L_cov, L_norm = model.heavy_path(features)
        omega = torch.linalg.eigvalsh(S)[:, -1].item()
        omega_history.append(omega)
        Omega_T = sum(omega_history) * 0.5
        d_J = torch.trace(S[0]).item()

    data = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "mode": MODE,
        "omega": omega,
        "Omega_T": Omega_T,
        "d_J": d_J,
        "BTC": prices["btc"],
        "ETH": prices["eth"],
        "SOL": prices["sol"]
    }
    
    # Write to log for dashboard to read
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
        
import argparse

async def main():
    parser = argparse.ArgumentParser(description="Q-SPINN Live Pipeline")
    parser.add_argument("--backfill", action="store_true", help="Run historical data backfill before streaming")
    parser.add_argument("--demo", action="store_true",
                        help="allow running with weights whose training contract does not match the live inputs")
    args = parser.parse_args()
    if WEIGHTS_ERROR:
        raise SystemExit(f"Refusing to run: {WEIGHTS_ERROR}")
    if not CONTRACT_OK and not args.demo:
        raise SystemExit("Refusing to run: training/live input contract mismatch on "
                         f"{CONTRACT_MISMATCH}. Retrain on matched live inputs, or pass --demo to run a "
                         "clearly labelled demonstration.")
    print(f"Mode: {MODE}")
    print("=== Launching Q-SPINN Production Pipeline ===")
    
    if args.backfill:
        print("Running historical data backfill...")
        from ingestor import backfill_historical
        await backfill_historical()
        print("Backfill complete.")
        
    print("Wiring ingestor WebSocket to PyTorch Q-SPINN Engine...")
    print(f"Outputting inference stream to {LOG_FILE}")
    
    # Clear previous log
    if LOG_FILE.exists():
        LOG_FILE.unlink()
        
    await stream_live(callback=process_tick)

if __name__ == "__main__":
    asyncio.run(main())
