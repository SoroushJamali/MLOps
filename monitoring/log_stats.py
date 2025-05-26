import os, json, logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
import yaml
from alert import send_slack, send_pagerduty
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CFG_PATH = Path(__file__).parent / "config.yaml"
cfg = yaml.safe_load(open(CFG_PATH, "r"))

TRAIN_PATH = Path(cfg["paths"]["train"])
PRED_PATH  = Path(cfg["paths"]["predict"])
REPORT     = Path(cfg["paths"]["report"])

MEAN_T = cfg["thresholds"]["mean_drift"]
KS_T   = cfg["thresholds"]["ks_pvalue"]

def load_df(p: Path):
    df = pd.read_parquet(p)
    return df

def summarize(df):
    stats = df.describe().T
    stats["nulls"] = df.isnull().sum()
    return stats

def detect_drift(train, pred):
    feats = train.columns.intersection(pred.columns)
    mt, kt = {}, {}
    for f in feats:
        mt[f] = abs(train[f].mean() - pred[f].mean())
        _, kt[f] = ks_2samp(train[f].dropna(), pred[f].dropna())
    return mt, kt

def main():
    train = load_df(TRAIN_PATH).drop(columns=["target","customerId","date"], errors="ignore")
    pred  = load_df(PRED_PATH).drop(columns=["customerId","date"], errors="ignore")

    stats_train = summarize(train)
    stats_pred  = summarize(pred)
    logger.info("=== Train ===\n%s", stats_train)
    logger.info("=== Predict ===\n%s", stats_pred)

    mean_drift, ks_p = detect_drift(train, pred)
    report = {"mean_drift": mean_drift, "ks_pvalue": ks_p}
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", REPORT)

    # Alerts
    drift_feats = [f for f,v in mean_drift.items() if v>MEAN_T] + \
                  [f for f,v in ks_p.items()    if v<KS_T]
    drift_feats = list(set(drift_feats))
    if drift_feats:
        msg = f"Drift detected for features: {drift_feats}"
        send_slack(msg)
        send_pagerduty(msg)
    else:
        logger.info("No drift detected.")

if __name__=="__main__":
    main()
