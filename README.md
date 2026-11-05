# Lightweight DDoS Detection

## Project Title

Lightweight DDoS Detection Using Minimal Feature Selection
and Computational Cost Analysis

## Overview

This project compares four machine-learning models for
network traffic classification:

- Decision Tree
- Random Forest
- Logistic Regression
- XGBoost

Mutual information is used to rank numerical features.
The models are compared using predictive performance
and computational cost.

The best model is saved and used by a Streamlit
frontend for CSV-based traffic classification.

## Project Structure

- data/: Training and testing CSV files
- feature_selection/: Feature-ranking script and results
- evaluation/: Model comparison script and results
- saved_models/: Best trained model and metadata
- frontend/: Streamlit application
- requirements.txt: Python dependencies

## Installation

Create a virtual environment:

    python -m venv .venv

Activate on Windows:

    .venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

## Run the Project

Generate the feature ranking:

    python feature_selection/generate_feature_ranking.py

Train and compare the models:

    python evaluation/compare_model.py

Start the Streamlit frontend:

    streamlit run frontend/app.py

## Dataset Requirements

Place the training and testing datasets in the data directory:

- train_dataset.csv
- test_dataset.csv

The target column is expected to be named Label.
If the dataset uses a different target column name,
update TARGET_COLUMN in the Python scripts.

The baseline implementation uses numerical features.

## Outputs

- feature_selection/information_gain_ranking.csv
- evaluation/model_comparison.csv
- saved_models/best_model.joblib
- saved_models/model_metadata.json

## Limitations

The application predicts labels for records in uploaded CSV files.
It does not capture live network traffic.

Model performance depends on the quality and representativeness
of the dataset. Test accuracy alone does not guarantee reliable
detection in a real deployment.