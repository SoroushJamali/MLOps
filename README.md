````md
# MLOps Assignment

## Overview
End-to-end pipeline: data ingestion → preprocessing → distributed training → batch inference → monitoring → evaluation → serving.

## Prerequisites

1. **Python 3.9+** & **pip** installed  
2. **Docker Desktop** installed and running  
3. **DVC** (Data Version Control) installed:  
   ```powershell
   pip install dvc
````

4. *(Optional)* **Minikube** or access to a Kubernetes cluster
5. *(Optional)* **PostgreSQL** running (or skip—scripts will fall back to local files)

## Quickstart (Windows)

Open **PowerShell** (not necessarily elevated) in `mlops-assignment/` and run:

1. **Create & activate virtualenv**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```

2. **Install Python dependencies**

   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure environment**

   * Update your `.env` file with local path
   * Fill in any credentials (DB, Slack/PagerDuty, MLflow URI)

4. **Pull data & model artifacts**

   ```powershell
   dvc pull            # downloads data/train, data/predict, models/ from remote
   ```

5. **Run distributed training**

   ```powershell
   python training/train.py
   ```

6. **Run batch inference**

   ```powershell
   python inference/predict.py
   ```

7. **Run data-drift monitoring**

   ```powershell
   python monitoring/log_stats.py
   ```

8. **Run model-performance evaluation**

   ```powershell
   python monitoring/evaluate_model.py
   ```

9. **Serve orchestration API**

   ```powershell
   uvicorn app.api:app --reload
   ```

---

## Docker

Run Docker container so you can run everything in isolation.

### 1. Build the image

From project root:

If you want to run only one container
```powershell
docker compose up --build -d
```
else
```powershell
docker build -t yourdockerhubuser/mlops-assignment:latest .
```

### 2. Run training in Docker

```powershell
docker run --rm `
  -v ${PWD}\data:C:\app\data `
  -v ${PWD}\models:C:\app\models `
  -v ${PWD}\.env:C:\app\.env `
  yourdockerhubuser/mlops-assignment:latest `
  python training/train.py
```

### 3. Run batch inference in Docker

```powershell
docker run --rm `
  -v ${PWD}\data:C:\app\data `
  -v ${PWD}\models:C:\app\models `
  -v ${PWD}\.env:C:\app\.env `
  yourdockerhubuser/mlops-assignment:latest `
  python inference/predict.py
```

### 4. Run monitoring & evaluation in Docker

```powershell
docker run --rm `
  -v ${PWD}\data:C:\app\data `
  -v ${PWD}\models:C:\app\models `
  -v ${PWD}\.env:C:\app\.env `
  yourdockerhubuser/mlops-assignment:latest `
  python monitoring/log_stats.py

docker run --rm `
  -v ${PWD}\data:C:\app\data `
  -v ${PWD}\models:C:\app\models `
  -v ${PWD}\.env:C:\app\.env `
  yourdockerhubuser/mlops-assignment:latest `
  python monitoring/evaluate_model.py
```

---

## Kubernetes

If you have a Kubernetes cluster (e.g., via Minikube), you can schedule the monitoring and evaluation scripts as CronJobs.

1. **Ensur your Docker image is pushed** to a registry:

   ```powershell
   docker push yourdockerhubuser/mlops-assignment:latest
   ```

2. **Apply the manifests**:

   ```powershell
   kubectl apply -f kubernetes/deployment.yaml       # if you have a long-running service
   kubectl apply -f kubernetes/service.yaml

   # Schedule data‐drift monitoring every hour
   kubectl apply -f kubernetes/job_monitoring.yaml

   # Schedule model evaluation daily at 1 AM
   kubectl apply -f kubernetes/cronjob-evaluate-model.yaml
   ```

3. **Verify CronJobs**:

   ```powershell
   kubectl get cronjob
   kubectl get jobs        # see recent runs
   kubectl logs <pod-name> # inspect job logs
   ```
 ```
mlops-assignment/
├── .gitignore                      # Files and directories to ignore in Git
├── README.md                       # Full project setup, usage, Docker/K8s instructions (this file)
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables for secrets and configuration (e.g., API keys, DB URIs)
│
├── data/                           # Raw and processed data for training and inference
│   ├── train/                      # Training datasets (e.g., chunk_01.parquet, chunk_02.parquet)
│   │   └── chunk_01.parquet
│   └── predict/                    # Prediction datasets (same format as training data)
│       └── chunk_01.parquet
│
├── training/                       # Components for the model training pipeline
│   ├── train.py                    # Main script for model training logic (designed for Dask for scalability)
│   ├── preprocess.py               # Scripts for feature engineering and data transformation
│   ├── pipeline.py  
│   └── config.yaml                 # Configuration file for training parameters (hyperparameters, feature lists, data paths)
│
├── inference/                      # Components for batch and real-time inference
│   ├── predict.py                  # Logic for running batch inference and saving predictions (e.g., to database or file)
│   └── config.yaml                   # FastAPI application for serving real-time model predictions
│
├── monitoring/                     # Scripts and configurations for data and model performance monitoring
│   ├── log_stats.py                # Compares current inference data against training data to detect data drift
│   ├── evaluate_model.py           # Evaluates model performance over time and logs metrics to MLflow
│   ├── monitor_config.yaml         # Configuration for monitoring thresholds and features to monitor
│   └── monitoring_report.json      # Generated report containing data drift analysis for inspection
│
├── models/                         # Directory for storing trained model artifacts
│   └── model.pkl                   # Example of a trained model artifact (can also be retrieved from MLflow)
│
├── notebooks/                      # Jupyter notebooks for exploratory data analysis and reports
│   ├── eda.ipynb                   # Exploratory Data Analysis and data understanding
│   └── drift_analysis.ipynb        # Optional notebook for visualizing and analyzing data drift manually
│
├── kubernetes/                     # Kubernetes manifests for deployment and scheduled jobs
│   ├── deployment.yaml             # Kubernetes Deployment specification for the inference API
│   ├── service.yaml                # Kubernetes Service and Ingress configuration to expose the inference API
│   ├── job_monitoring.yaml         # Kubernetes Job to run `log_stats.py` for batch data monitoring
│   └── cronjob-evaluate-model.yaml # Kubernetes CronJob to periodically run `evaluate_model.py` for performance monitoring
│
├── tests/                          # Unit and integration tests for various components
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_monitoring.py
│
└── app/                            # Main application layer, often containing an API for orchestration
    └── api.py    
 ```      

# ML Prediction API

This project serves as a minimal API layer for serving a trained machine learning model. It provides endpoints for health checks and predicting propensity scores based on input features.

---

## How to Run and Test

### 1. Ensure a Model is Trained

Before running the API, make sure you've trained a model using the provided training pipeline:

```bash
python training/train.py
```

This will generate a `.pkl` model file in the `models/` directory.

---

### 2. Start the API Server

Launch the API server by running:

```bash
python app/api.py
```

You should see output indicating the server is running (typically at `http://0.0.0.0:8000`) and which model file it has loaded.

---

### 3. Test the Prediction Endpoint

Open a **new terminal** and run the following command to send a sample prediction request:

```bash
curl -X POST http://localhost:8000/predict_propensity \
     -H "Content-Type: application/json" \
     -d '{"feature1": 50.1, "feature2": 0.9}'
```

You should receive a JSON response similar to:

```json
{"propensity_to_buy": 0.8707384467124939}
```

**Windows Users:**
If you're using Command Prompt or PowerShell, you may need to escape quotes like this:

```bash
curl -X POST http://localhost:8000/predict_propensity -H "Content-Type: application/json" -d "{\"feature1\": 50.1, \"feature2\": 0.9}"
```

---

### 4. Test the Health Check Endpoint

Check the API's health and whether a model is successfully loaded:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "<path-to-your-model.pkl>"
}
```



# Testing

The `tests/` directory contains automated tests to ensure the reliability and correctness of our machine learning pipeline components. We use `pytest` as our testing framework for its simplicity and effectiveness.

Currently, the primary focus is on validating the core training pipeline.

---

## Test Structure

- `test_training.py`: Contains tests specifically designed to verify the functionality of the model training process.

---

## How to Run Tests

To execute all tests, navigate to the project's root directory and run:

```bash
pytest tests/


To run tests for just the `test_training.py` file:

```bash
pytest tests/test_training.py
```

---

## What the Tests Verify (from `test_training.py`)

### `test_build_pipeline_end_to_end()` (Unit Test)

* Tests the `build_pipeline()` function defined in `training/pipeline.py`.
* Creates a small, in-memory DataFrame as sample data.
* Verifies that:

  * The function returns an `sklearn.pipeline.Pipeline` instance.
  * The pipeline can be fit to data and used for `.predict()` and `.predict_proba()`.
  * Predictions are binary (0 or 1).
  * Probabilities are valid and sum to 1.

### `test_training_script_creates_model()` (Integration Test)

* Simulates a full run of the `training/train.py` script.
* Uses `pytest` fixtures (`tmp_path`, `monkeypatch`) to create an isolated environment:

  * Generates a temporary `.parquet` data file.
  * Clears the `models/` directory to ensure a fresh run.
  * Calls the `main()` function in `train.py`.

#### Assertions:

* Confirms exactly one new model artifact (`pipeline_*.pkl`) is created in the `models/` directory.
* Loads the model using `joblib`.
* Ensures the loaded pipeline can make predictions using `predict_proba` on sample data, verifying model integrity and usability.

