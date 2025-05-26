import sys
from pathlib import Path

# Add project root to Python path so "import training" works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
import shutil

import training.train as train_module
from training.pipeline import build_pipeline

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR   = PROJECT_ROOT / "data" / "train"

def test_build_pipeline_end_to_end():
    """Unit-test for the sklearn Pipeline."""
    df = pd.DataFrame({
        "feature1": [1.0, np.nan, 3.0, 4.0],
        "feature2": [0.1, 0.2, 0.3, 0.4],
        "target":   [0,    1,     0,    1]
    })
    X = df[["feature1", "feature2"]]
    y = df["target"]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert preds.shape == (4,)
    assert set(preds).issubset({0,1})

    probs = pipeline.predict_proba(X)
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-6)

def test_training_script_creates_model(tmp_path, monkeypatch):
    """
    Integration-style test: run training.train.main() and verify
    that it writes a model artifact into the project’s models/ folder.
    """
    # 1. Prepare a small parquet in data/train
    data_train = DATA_DIR
    data_train.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "feature1": [1.0,2.0,3.0,4.0],
        "feature2": [0.5,0.6,0.7,0.8],
        "target":   [0,1,0,1],
        "customerId": ["c1","c2","c3","c4"],
        "date": pd.date_range("2025-01-01", periods=4)
    })
    file_path = data_train / "chunk_01.parquet"
    df.to_parquet(file_path)

    # 2. Clean out models/
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Run training
    train_module.main()

    # 4. Assert exactly one new model file was created
    models = list(MODELS_DIR.glob("pipeline_*.pkl"))
    assert len(models) == 1, f"Expected one model artifact, found {len(models)}"

    # 5. Load and verify
    pipeline = joblib.load(models[0])
    sample = df[["feature1","feature2"]].iloc[:2]
    probs = pipeline.predict_proba(sample)
    assert probs.shape == (2, 2)
