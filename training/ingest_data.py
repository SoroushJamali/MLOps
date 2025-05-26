import argparse
import pandas as pd
from pathlib import Path

def main(input_dir: str, output_dir: str):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in input_dir.glob("*.parquet"):
        df = pd.read_parquet(f)
        # → place any domain‐specific cleaning/transforms here
        df.to_parquet(output_dir / f.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
