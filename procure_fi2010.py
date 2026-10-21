import sys
import shutil
import zipfile
import pandas as pd
from pathlib import Path

# Paths
REFERENCE_DIR = Path("data/reference/fi2010")
ZIP_PATH = REFERENCE_DIR / "BenchmarkDatasets.zip"

def main(setup2=False, segments=None, allow_heuristic=False):
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not ZIP_PATH.exists():
        print(f"Error: The FI-2010 dataset was not found at {ZIP_PATH}.")
        print("\n" + "="*50)
        print("              MANUAL DOWNLOAD REQUIRED")
        print("="*50)
        print("Because the official FI-2010 dataset is a 1.86 GB archive hosted on ")
        print("the Fairdata Etsin academic portal (which requires navigating their ")
        print("web interface), it cannot be downloaded automatically without a key.")
        print("\nPlease follow these steps to securely fetch it:")
        print("1. Go to: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649")
        print("2. Navigate to the 'Data' tab and download 'BenchmarkDatasets.zip'.")
        print(f"3. Move the downloaded ZIP into this exact folder:")
        print(f"   -> {REFERENCE_DIR.absolute()}")
        print("4. Run this script again.")
        print("="*50 + "\n")
        sys.exit(1)
        
    print(f"Found FI-2010 dataset at {ZIP_PATH}!")
    
    # 1. Create a temp directory for extraction
    temp_dir = REFERENCE_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        print("Extracting outer ZIP...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extract("published/BenchmarkDatasets/BenchmarkDatasets.zip", temp_dir)
            
        inner_zip = temp_dir / "published" / "BenchmarkDatasets" / "BenchmarkDatasets.zip"
        
        print("Extracting inner ZIP...")
        with zipfile.ZipFile(inner_zip, 'r') as z:
            # Extract just the NoAuction Zscore folder
            for member in z.namelist():
                if "NoAuction/1.NoAuction_Zscore/" in member:
                    z.extract(member, temp_dir)
                    
        if setup2:
            _build_setup2(temp_dir, segments, allow_heuristic)
            return

        # 2. Parse the files
        print("Parsing folds and converting to Parquet...")
        base_path = temp_dir / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"
        
        all_dfs = []
        for split in ["Train", "Test"]:
            split_folder = base_path / f"NoAuction_Zscore_{'Training' if split == 'Train' else 'Testing'}"
            for fold in range(1, 10):
                prefix = 'Train_Dst' if split == 'Train' else 'Test_Dst'
                file_path = split_folder / f"{prefix}_NoAuction_ZScore_CF_{fold}.txt"
                
                if not file_path.exists():
                    continue
                    
                print(f"  Reading {split} fold {fold}...")
                # FI-2010 has 149 rows (144 features, 5 labels), and cols = timesteps
                df = pd.read_csv(file_path, sep=r'\s+', header=None)
                
                # Transpose so rows = timesteps, cols = features
                df = df.T
                
                # Add metadata columns
                df['split'] = split
                df['fold'] = fold
                
                all_dfs.append(df)
                
        print("Concatenating into single DataFrame...")
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # 3. Rename columns for clarity (features 0-143, labels 144-148)
        col_names = {i: f"feature_{i}" for i in range(144)}
        col_names.update({
            144: "label_k10",
            145: "label_k20",
            146: "label_k30",
            147: "label_k50",
            148: "label_k100"
        })
        final_df = final_df.rename(columns=col_names)
        
        # Convert labels from 1/2/3 to 0/1/2 (standard for PyTorch CrossEntropy)
        label_cols = ["label_k10", "label_k20", "label_k30", "label_k50", "label_k100"]
        final_df[label_cols] = final_df[label_cols] - 1.0
        
        # 4. Save to Parquet
        out_parquet = REFERENCE_DIR / "fi2010_noauction_zscore.parquet"
        print(f"Saving to {out_parquet}...")
        final_df.to_parquet(out_parquet, engine='pyarrow', compression='zstd')
        
        print("Success! FI-2010 successfully procured and normalized.")
        
    finally:
        # Cleanup
        print("Cleaning up scratch files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

def _build_setup2(temp_dir, segments=None, allow_heuristic=False):
    """Setup 2: Train CF7 once + Test CF7/8/9, with recovered stock/day segments (see experiments/fi2010_setup2.py)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
    import fi2010_setup2 as f2
    base = temp_dir / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"
    read = lambda p: pd.read_csv(p, sep=r'\s+', header=None).T.values
    train = read(base / "NoAuction_Zscore_Training" / "Train_Dst_NoAuction_ZScore_CF_7.txt")
    tests = {name: read(base / "NoAuction_Zscore_Testing" / f"Test_Dst_NoAuction_ZScore_CF_{cf}.txt")
             for name, cf in f2.TEST_DAYS.items()}
    import json
    lengths = json.loads(Path(segments).read_text()) if segments else None
    frames = f2.build_setup2_frames(train, tests, segment_lengths=lengths, allow_heuristic=allow_heuristic)
    f2.save_setup2(frames, REFERENCE_DIR)
    for name, r in f2.audit(frames).items():
        flag = "AUTHORITATIVE" if r["confident"] else "HEURISTIC-UNVERIFIED"
        print(f"  [{flag}] {name}: rows={r['rows']:,} segments={r['n_segments']}")
    print(f"Wrote {REFERENCE_DIR/'fi2010_setup2.parquet'} and segment_audit.json - INSPECT the audit before training.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup2", action="store_true")
    ap.add_argument("--segments", default=None, help="JSON {partition: [per-stock/day row counts]} (authoritative)")
    ap.add_argument("--allow-heuristic", action="store_true", help="exploration only; training will refuse the result")
    a = ap.parse_args()
    main(setup2=a.setup2, segments=a.segments, allow_heuristic=a.allow_heuristic)
