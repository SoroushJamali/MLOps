import yaml
import joblib
import mlflow
from pathlib import Path
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from mlflow.models import infer_signature
from training.pipeline import build_pipeline
import multiprocessing

def main():
    # ——— 0) Paths & Dask config ————————————————————————————————————
    ROOT       = Path(__file__).parent.parent
    DATA_DIR   = ROOT / "data" / "train"
    MODELS_DIR = ROOT / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))

    # ——— 1) Start Dask client —————————————————————————————————————
    client = Client(
        address=cfg["dask"]["scheduler_address"],
        n_workers=cfg["dask"]["n_workers"],
        threads_per_worker=cfg["dask"]["threads_per_worker"]
    )

    # ——— 2) Read parquet files in parallel (Windows-safe) ——————————
    file_list = [str(p) for p in DATA_DIR.glob("*.parquet")]
    if not file_list:
        raise FileNotFoundError(f"No .parquet files found in {DATA_DIR}")
    df = dd.read_parquet(file_list).compute()

    # ——— 3) Prepare features/labels ————————————————————————————
    exclude = {"target", "customerId", "date"}
    feature_cols = sorted(set(df.columns) - exclude)
    X = df[feature_cols]
    y = df["target"]

    # ——— 4) Split & train —————————————————————————————————————
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=cfg["model"]["random_seed"]
    )

    pipeline = build_pipeline()
    mlflow.set_experiment("propensity_model")
    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)[:, 1]

        mlflow.log_params(cfg["model"]["xgb_params"])
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_val, preds),
            "auc": roc_auc_score(y_val, probs)
        })

        sig = infer_signature(X_val.head(), pipeline.predict(X_val.head()))
        mlflow.sklearn.log_model(pipeline, "model", signature=sig)

        out = MODELS_DIR / f"pipeline_{run.info.run_id}.pkl"
        joblib.dump(pipeline, out)

    print("✅ Training complete.")

if __name__ == "__main__":
    # On Windows, this protects child-process startup
    multiprocessing.freeze_support()
    main()
