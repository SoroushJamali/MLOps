import yaml
import joblib
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, exc
import logging
# ——— 0) Load config ————————————————————————————————————————
CFG_PATH = Path(__file__).parent / "config.yaml"
cfg = yaml.safe_load(open(CFG_PATH, "r"))

MODEL_PATH = cfg["model"]["path"]
DB_URI     = cfg["database"].get("uri")
DB_TABLE   = cfg["database"].get("table", "predictions")

# ——— 1) Load model ———————————————————————————————————————
pipeline = joblib.load(MODEL_PATH)
logging.info(f"Loaded model from {MODEL_PATH}")

# ——— 2) Load & concat prediction data —————————————————————————
DATA_DIR = Path(__file__).parent.parent / "data" / "predict"
files = list(DATA_DIR.glob("*.parquet"))
if not files:
    raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")
df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

# ——— 3) Prepare features & predict ———————————————————————————
drop_cols = [c for c in ("customerId", "date", "target") if c in df.columns]
X = df.drop(columns=drop_cols, errors="ignore")
df["propensity_to_buy"] = pipeline.predict_proba(X)[:, 1]

# ——— 4) Attempt to write to Postgres (fallback if unavailable) —————
def write_to_db(engine_uri, table, data):
    engine = create_engine(engine_uri)
    with engine.begin() as conn:
        data.to_sql(table, conn, if_exists="append", index=False)
    logging.info(f"✅ Written {len(data)} rows to table '{table}'")

def write_to_parquet(path: Path, data: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)
    logging.info(f"✅ Written {len(data)} rows to Parquet at {path}")

try:
    if DB_URI:
        write_to_db(DB_URI, DB_TABLE, df[["customerId", "propensity_to_buy"]])
    else:
        raise ValueError("No database URI configured")
except (exc.SQLAlchemyError, ConnectionError, ValueError) as e:
    logging.info(f"⚠️  DB write failed ({e}), falling back to local file.")
    out_path = Path(__file__).parent.parent / "inference_results.parquet"
    write_to_parquet(out_path, df[["customerId", "propensity_to_buy"]])

logging.info("Batch inference complete.")
