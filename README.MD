# AutoML Project

An end-to-end AutoML platform that trains multiple models, compares performance with a leaderboard, generates reports, and provides a Streamlit UI.

## Features

- Automatic data ingestion and validation
- Preprocessing (missing values, encoding, train-test split)
- Feature engineering (polynomial features + feature selection)
- Multi-model training for classification and regression
- Hyperparameter tuning
- Model leaderboard and best model selection
- Evaluation metrics and bias detection
- Explainability (feature importance)
- Report generation and model persistence
- Streamlit web application for interactive usage

## Project Structure

```text
AutoML project/
├── app.py
├── configs/
├── data/
│   └── raw/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── pipeline.py
│   ├── model_selection.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   └── ...
└── README.MD
```

## Installation

1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

### 1) Run pipeline from terminal

```bash
python -m src.pipeline
```

### 2) Run Streamlit app

```bash
streamlit run app.py
```

## Input Data

- Supported formats: `.csv`, `.xlsx`
- Place files under `data/raw/` or upload from Streamlit
- Select target column from UI before running pipeline

## Outputs

- Trained model files in `models/`
- EDA plots and reports in `reports/`
- Text report: `reports/model_report.txt`
- Leaderboard shown in CLI and Streamlit

## Core Modules

- `src/pipeline.py` - orchestration flow
- `src/data_ingestion.py` - loading and validation
- `src/preprocessing.py` - cleaning and splitting
- `src/feature_engineering.py` - feature generation/selection
- `src/model_selection.py` - model training and scoring
- `src/hyperparameter_tuning.py` - search tuning
- `src/evaluation.py` - evaluation metrics
- `src/model_leaderboard.py` - ranking models
- `src/report_engine.py` - report generation
- `src/eda.py` - EDA plots and analysis

## Notes

- If target leakage handling is enabled in pipeline, keep target excluded during feature generation.
- For heavy models (XGBoost/LightGBM), first run may take longer.
