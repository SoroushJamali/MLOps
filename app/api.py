from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from pathlib import Path
import logging
import numpy as np  # Import numpy

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Propensity to Buy Prediction API",
    description="API for predicting customer propensity to buy using a trained ML model."
)

# --- Model Loading Logic ---
# Define the directory where models are stored
MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = None
model = None


def load_latest_model():
    """
    Finds and loads the most recently modified .pkl model file from the MODELS_DIR.
    This assumes that the latest modified file is the desired model.
    """
    global model, MODEL_PATH
    try:
        # List all .pkl files in the models directory
        model_files = list(MODELS_DIR.glob("*.pkl"))
        if not model_files:
            logger.warning(f"No .pkl model files found in {MODELS_DIR}. Please train a model first.")
            return None

        # Sort files by modification time (most recent first)
        model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest_model_path = model_files[0]

        logger.info(f"Attempting to load model from: {latest_model_path}")
        loaded_model = joblib.load(latest_model_path)

        MODEL_PATH = latest_model_path  # Store the path of the loaded model
        logger.info(f"Successfully loaded model from: {MODEL_PATH}")
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load model from {MODELS_DIR}: {e}")
        return None


# Load the model when the application starts up
model = load_latest_model()


# --- Pydantic Model for Request Body ---
class PredictionFeatures(BaseModel):
    feature1: float
    feature2: float
    # Add any other features your model was trained on, e.g.:
    # feature3: float
    # feature4: float


# --- Prediction Endpoint ---
@app.post("/predict_propensity")
def predict_propensity(features: PredictionFeatures):
    """
    Predicts the propensity to buy for a given set of customer features.
    """
    if model is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded. Please ensure a model is trained and available.")

    input_df = pd.DataFrame([features.dict()])

    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[:, 1][0]
        else:
            prob = model.predict(input_df)[0]
            logger.warning("Model does not have 'predict_proba'. Returning direct prediction.")

        # --- FIX: Convert NumPy float to standard Python float ---
        propensity_to_buy = float(prob)

        return {"propensity_to_buy": propensity_to_buy}
    except KeyError as ke:
        logger.error(f"Missing feature in input data: {ke}. Ensure all required features are provided.")
        raise HTTPException(status_code=400,
                            detail=f"Missing required feature(s) in input: {ke}. Check your request body.")
    except Exception as e:
        logger.error(f"Prediction failed due to an unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")


# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running and model is loaded.
    """
    if model is not None:
        return {"status": "healthy", "model_loaded": True, "model_path": str(MODEL_PATH)}
    else:
        return {"status": "unhealthy", "model_loaded": False}


# --- Main entry point for Uvicorn ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)