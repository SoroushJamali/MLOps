import os
import json
import logging
from pathlib import Path

import yaml
import pandas as pd
import joblib
from sqlalchemy import create_engine, exc
from sklearn.metrics import accuracy_score, roc_auc_score
from dotenv import load_dotenv

# Optional import; if mlflow isn’t installed or server is down, we’ll catch errors
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ——— Load environment & config —————————————————————————————————
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("model_evaluator")

cfg = yaml.safe_load(open(Path(__file__).parent / "monitor_config.yaml"))

# DB settings
DB_CONF       = cfg["database"]
DB_URI        = os.getenv("DB_URI", DB_CONF.get("uri"))
PRED_TABLE    = DB_CONF.get("predictions_table", "predictions")
ACTUALS_TABLE = DB_CONF.get("actuals_table", "actuals")

# Local fallback paths
FALLBACK_PRED_PATH    = Path(__file__).parent.parent / "inference_results.parquet"
FALLBACK_ACTUALS_DIR  = Path(__file__).parent.parent / "data" / "predict"

# Performance thresholds
ACCURACY_THRESHOLD = cfg["performance"]["min_accuracy"]
AUC_THRESHOLD      = cfg["performance"]["min_auc"]

# MLflow setup (might fail)
if MLFLOW_AVAILABLE:
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("model_performance")
        logger.info("MLflow tracking configured.")
    except Exception as e:
        logger.warning("MLflow unavailable (%s); will fallback to local JSON", e)
        MLFLOW_AVAILABLE = False

def load_from_db():
    engine = create_engine(DB_URI)
    preds   = pd.read_sql_table(PRED_TABLE, engine)
    actuals = pd.read_sql_table(ACTUALS_TABLE, engine)
    df = preds.merge(actuals, on="customerId", how="inner",
                     suffixes=("_pred","_true"))
    logger.info("Loaded %d records from DB", len(df))
    return df

def load_from_local():
    if not FALLBACK_PRED_PATH.exists():
        raise FileNotFoundError(f"No predictions at {FALLBACK_PRED_PATH}")
    preds = pd.read_parquet(FALLBACK_PRED_PATH)
    files = list(FALLBACK_ACTUALS_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No actuals in {FALLBACK_ACTUALS_DIR}")
    actuals = pd.concat((pd.read_parquet(f) for f in files),
                        ignore_index=True)[["customerId","target"]]
    actuals = actuals.rename(columns={"target":"actual_label"})
    df = preds.merge(actuals, on="customerId", how="inner")
    df = df.rename(columns={"propensity_to_buy":"propensity_to_buy_pred"})
    logger.info("Loaded %d records from local files", len(df))
    return df

def evaluate(df: pd.DataFrame):
    y_true = df["actual_label"]
    y_pred = df["propensity_to_buy_pred"]
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    auc = roc_auc_score(y_true, y_pred)
    logger.info("Evaluation — accuracy: %.4f, AUC: %.4f", acc, auc)
    return acc, auc

def log_metrics_mlflow(acc, auc):
    try:
        with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("auc", auc)
        logger.info("Logged metrics to MLflow.")
    except Exception as e:
        logger.warning("Failed to log to MLflow (%s); skipping.", e)

def write_fallback_report(acc, auc):
    report = {
        "accuracy": acc,
        "auc": auc,
        "thresholds": {
            "accuracy": ACCURACY_THRESHOLD,
            "auc": AUC_THRESHOLD
        }
    }
    out = Path(__file__).parent / "performance_report.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote local performance report to %s", out)

def alert_if_needed(acc, auc):
    msgs = []
    if acc < ACCURACY_THRESHOLD:
        msgs.append(f"Accuracy {acc:.3f} < {ACCURACY_THRESHOLD}")
    if auc < AUC_THRESHOLD:
        msgs.append(f"AUC {auc:.3f} < {AUC_THRESHOLD}")
    if msgs:
        logger.warning("Performance alert: %s", "; ".join(msgs))

def main():
    # 1) Load data
    try:
        df = load_from_db() if DB_URI else (_ for _ in ()).throw(ValueError("No DB_URI"))
    except Exception as e:
        logger.warning("DB load failed (%s); using local", e)
        df = load_from_local()

    if df.empty:
        logger.error("No data to evaluate; exiting.")
        return

    # 2) Evaluate
    acc, auc = evaluate(df)

    # 3) Log metrics
    if MLFLOW_AVAILABLE:
        log_metrics_mlflow(acc, auc)
    else:
        write_fallback_report(acc, auc)

    # 4) Alert
    alert_if_needed(acc, auc)

if __name__ == "__main__":
    main()
