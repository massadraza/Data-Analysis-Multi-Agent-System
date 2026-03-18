# ML System Architecture

## Overview
Production-ready end-to-end ML system for Iris classification using RandomForest, served via FastAPI with a Streamlit frontend.

## Directory Structure
```
ml_system/
├── src/
│   ├── data/          # Data loading & preprocessing
│   ├── models/        # Model training & serialization
│   └── evaluation/    # Metrics & evaluation
├── api/               # FastAPI backend
├── frontend/          # Streamlit UI
├── scripts/           # Training & utility scripts
├── artifacts/         # Trained models & metrics (gitignored)
├── tests/
│   ├── unit/          # Unit tests for model & data
│   ├── integration/   # API integration tests
│   └── fixtures/      # Test data & fixtures
├── .github/workflows/ # CI/CD
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Components

### 1. ML Pipeline (`src/`)
- **Data** (`src/data/preprocessing.py`): Load Iris dataset, train/test split, feature scaling
- **Models** (`src/models/trainer.py`): RandomForestClassifier training, hyperparameter config, joblib serialization
- **Evaluation** (`src/evaluation/metrics.py`): Accuracy, classification report, confusion matrix → `artifacts/metrics.json`
- **Training Script** (`scripts/train.py`): End-to-end training entrypoint

### 2. FastAPI Backend (`api/`)
- `api/main.py`: FastAPI app
- `GET /health` — returns `{"status": "healthy"}`
- `POST /predict` — accepts `{features: [f1, f2, f3, f4]}`, returns `{prediction, probability}`
- `GET /metrics` — returns model evaluation metrics from `artifacts/metrics.json`
- Loads model from `artifacts/model.pkl` at startup

### 3. Streamlit Frontend (`frontend/`)
- `frontend/app.py`: Interactive UI
- Prediction form with 4 numeric inputs (sepal/petal length/width)
- Model metrics dashboard with accuracy and confusion matrix display
- Communicates with FastAPI backend via HTTP

### 4. DevOps
- `Dockerfile`: Multi-stage build for API
- `docker-compose.yml`: Services for API + frontend
- `.github/workflows/ci.yml`: Lint, test, build on push/PR

### 5. Testing
- Unit tests: model training, data preprocessing, evaluation
- Integration tests: API endpoint testing with test client
- Fixtures: sample data, mock model artifacts

## Data Flow
```
Iris Dataset → Preprocessing → Train RandomForest → Save model.pkl + metrics.json
                                                          ↓
User Input → Streamlit → FastAPI /predict → Load model.pkl → Prediction → Response
```

## Configuration
- Model hyperparameters in `src/models/config.py`
- API settings via environment variables
- All artifacts stored in `artifacts/` directory
