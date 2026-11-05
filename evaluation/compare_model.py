from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")


# Project structure: evaluation/compare_models.py, models/, train_dataset.csv, test_dataset.csv
PROJECT_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "evaluation" else Path.cwd()
MODELS_DIR = PROJECT_DIR / "models"
TRAIN_FILE = PROJECT_DIR / "data/train_dataset.csv"
TEST_FILE = PROJECT_DIR / "data/test_dataset.csv"
RANKING_CANDIDATES = [
    PROJECT_DIR / "feature_selection" / "information_gain_ranking.csv",
    PROJECT_DIR / "information_gain_ranking.csv",
    MODELS_DIR / "information_gain_ranking.csv",
]
RESULT_DIR = PROJECT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_COUNTS = [3, 5, 7, 10, 15, 20, 25, 35, 39]
RANDOM_STATE = 42

# Original source file locations. These are checked so the run cannot silently
SOURCE_FILES = {
    "Decision Tree": MODELS_DIR / "decision_tree_model.py",
    "Random Forest": MODELS_DIR / "Random_Forest.ipynb",
    "Logistic Regression": MODELS_DIR / "logistic regression code.ipynb",
    "XGBoost": MODELS_DIR / "xgboost_ddos.py",
}


def check_sources():
    print("\nChecking original model source files:")
    for name, path in SOURCE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing source for {name}: {path}\n"
                "Update SOURCE_FILES near the top of this script to match your exact filenames."
            )
        print(f"  OK - {name}: {path.relative_to(PROJECT_DIR)}")
        # Read the notebook source so we verify the original notebook is valid JSON.
        if path.suffix == ".ipynb":
            with path.open("r", encoding="utf-8") as f:
                notebook = json.load(f)
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            )
            if not code.strip():
                raise ValueError(f"No code cells found in {path}")
    # Read the XGBoost script and ensure the expected original configuration is present.
    xgb_source = SOURCE_FILES["XGBoost"].read_text(encoding="utf-8")
    for setting in ("N_ESTIMATORS = 200", "MAX_DEPTH = 6", "LEARNING_RATE = 0.1", "SUBSAMPLE = 0.8", "COLSAMPLE_BYTREE = 0.8"):
        if setting not in xgb_source:
            raise ValueError(f"Expected XGBoost setting not found in original xgboost_ddos.py: {setting}")


def load_data():
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Could not find train_dataset.csv and/or test_dataset.csv.\n"
            f"Expected training file: {TRAIN_FILE}\nExpected test file: {TEST_FILE}\n"
            "Place this script in the evaluation folder inside your project, or update PROJECT_DIR."
        )
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    train.columns = train.columns.astype(str).str.strip()
    test.columns = test.columns.astype(str).str.strip()
    if "Label" not in train.columns or "Label" not in test.columns:
        raise ValueError("Both CSV files must contain a 'Label' target column.")
    # Matches the notebook preprocessing: infinities/missing values -> 0.
    train = train.replace([np.inf, -np.inf], np.nan).fillna(0)
    test = test.replace([np.inf, -np.inf], np.nan).fillna(0)
    train["Label"] = train["Label"].astype(str).str.strip()
    test["Label"] = test["Label"].astype(str).str.strip()
    return train, test


def load_ranking(train):
    ranking_path = next((p for p in RANKING_CANDIDATES if p.exists()), None)
    features = [c for c in train.columns if c != "Label"]
    if ranking_path:
        print(f"\nUsing existing Information Gain ranking: {ranking_path}")
        rank_df = pd.read_csv(ranking_path)
        feature_col = next((c for c in rank_df.columns if str(c).strip().lower().replace(" ", "_") in
                            {"feature", "features", "feature_name", "featurename", "attribute", "variable", "name"}), rank_df.columns[0])
        ranked = [str(x).strip() for x in rank_df[feature_col].dropna().tolist()]
        ranked = [x for x in dict.fromkeys(ranked) if x in features]
        if not ranked:
            raise ValueError(f"No matching feature names found in {ranking_path}")
        # If a pre-existing ranking is incomplete, append remaining dataset features.
        ranked += [f for f in features if f not in ranked]
    else:
        print("\nNo saved ranking CSV found. Calculating Information Gain on up to 100,000 rows...")
        sample = train.sample(n=min(100000, len(train)), random_state=RANDOM_STATE)
        X_sample = sample[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        enc = LabelEncoder()
        y_sample = enc.fit_transform(sample["Label"])
        scores = mutual_info_classif(X_sample, y_sample, random_state=RANDOM_STATE)
        ranked = pd.DataFrame({"Feature": features, "Information_Gain": scores}).sort_values(
            "Information_Gain", ascending=False
        )["Feature"].tolist()
        ranking_path = PROJECT_DIR / "feature_selection" / "information_gain_ranking.csv"
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Feature": ranked, "Rank": range(1, len(ranked) + 1)}).to_csv(ranking_path, index=False)
        print(f"Saved ranking to: {ranking_path}")
    return ranked


def build_model(model_name, n_classes):
    # These estimator definitions and settings match the original model sources.
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    if model_name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost") from e
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def main():
    print("=" * 75)
    print("DDoS MODEL COMPARISON: ACCURACY + TRAINING/PREDICTION COST")
    print("=" * 75)
    check_sources()
    train, test = load_data()
    print(f"\nTrain shape: {train.shape}; Test shape: {test.shape}")

    # Keep the original notebook's LabelEncoder approach.
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["Label"])
    unseen = sorted(set(test["Label"].unique()) - set(encoder.classes_))
    if unseen:
        raise ValueError(f"Test data contains labels absent from training data: {unseen}")
    y_test = encoder.transform(test["Label"])

    ranked_features = load_ranking(train)
    feature_counts = sorted(set(min(k, len(ranked_features)) for k in FEATURE_COUNTS))
    model_names = ["Decision Tree", "Random Forest", "Logistic Regression", "XGBoost"]
    rows = []

    for k in feature_counts:
        selected = ranked_features[:k]
        print(f"\n{'-' * 70}\nTop-{k} features: {selected}\n{'-' * 70}")
        X_train = train[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = test[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        for model_name in model_names:
            print(f"Training {model_name} with {k} features...", flush=True)
            model = build_model(model_name, len(encoder.classes_))
            start = time.perf_counter()
            if model_name == "XGBoost":
                model.fit(X_train, y_train, verbose=False)
            else:
                model.fit(X_train, y_train)
            training_time = time.perf_counter() - start

            start = time.perf_counter()
            predictions = model.predict(X_test)
            prediction_time = time.perf_counter() - start
            row = {
                "Model": model_name,
                "Number_of_Features": k,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision_Weighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "Recall_Weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "F1_Weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
                "Training_Time_s": training_time,
                "Prediction_Time_s": prediction_time,
                "Prediction_Time_per_Sample_us": prediction_time / len(y_test) * 1e6,
            }
            rows.append(row)
            print(f"  Accuracy={row['Accuracy']:.4f} | train={training_time:.2f}s | predict={prediction_time:.2f}s")

            # Release large model object before training the next model.
            del model

        # Save incrementally so progress is not lost if a long run is interrupted.
        interim = pd.DataFrame(rows)
        interim.to_csv(RESULT_DIR / "model_comparison_results.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(RESULT_DIR / "model_comparison_results.csv", index=False)
    summary = results.sort_values(["Number_of_Features", "Accuracy"], ascending=[True, False])
    summary.to_csv(RESULT_DIR / "model_comparison_sorted.csv", index=False)
    print("\nCompleted. Results saved to:")
    print(f"  {RESULT_DIR / 'model_comparison_results.csv'}")
    print(f"  {RESULT_DIR / 'model_comparison_sorted.csv'}")
    print("\nResults preview:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

