# ============================================================
# Project: Lightweight DDoS Detection Using Minimal Feature
#          Selection and Computational Cost Analysis
# Algorithm: XGBoost
# Dataset: CICIoT2023
# Feature Ranking: Information Gain
# ============================================================

import os
import time
import warnings
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RANKING_FILE = os.path.join(
    BASE_DIR, "information_gain_ranking.csv"
)

TRAIN_FILE = os.path.join(
    BASE_DIR, "train_dataset.csv"
)

TEST_FILE = os.path.join(
    BASE_DIR, "test_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "xgboost_models"
)

RESULTS_FILE = os.path.join(
    BASE_DIR, "xgboost_results.csv"
)

TARGET_COLUMN = "Label"

FEATURE_SUBSETS = [3, 5, 7, 10, 15, 20, 39]

RANDOM_STATE = 42

# XGBoost settings
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

# Use all available CPU cores.
N_JOBS = -1

# Create output folder.
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column_case_insensitive(dataframe, target_name):
    """
    Find a dataframe column without depending on capitalization.
    Return its actual column name.
    """
    for column in dataframe.columns:
        if str(column).strip().lower() == target_name.lower():
            return column

    return None


def load_feature_ranking(filepath):
    """
    Load feature ranking and identify the feature-name column.

    Supports common column names such as:
    Feature, Features, Feature_Name, and similar names.

    If no recognizable column exists, the first CSV column
    is used as a fallback.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Feature ranking file not found: {filepath}"
        )

    ranking_df = pd.read_csv(filepath)

    if ranking_df.empty:
        raise ValueError(
            "The information gain ranking file is empty."
        )

    print("\nRanking CSV columns:")
    print(ranking_df.columns.tolist())

    preferred_names = [
        "feature",
        "features",
        "feature_name",
        "featurename",
        "attribute",
        "variable",
        "name"
    ]

    feature_column = None

    for column in ranking_df.columns:
        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized in preferred_names:
            feature_column = column
            break

    if feature_column is None:
        # Fallback: use the first column.
        feature_column = ranking_df.columns[0]
        print(
            "\nWarning: Could not identify a named feature column."
        )
        print(
            f"Using the first column: {feature_column}"
        )

    features = (
        ranking_df[feature_column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Remove duplicate feature names while preserving order.
    features = list(dict.fromkeys(features))

    print(f"\nTotal ranked features: {len(features)}")

    return features


def clean_dataset(dataframe, dataset_name):
    """
    Normalize column whitespace, validate the target column,
    and convert feature values to numeric values.
    """
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    actual_target = find_column_case_insensitive(
        dataframe, TARGET_COLUMN
    )

    if actual_target is None:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' was not found in "
            f"{dataset_name}.\n"
            f"Available columns: {dataframe.columns.tolist()}"
        )

    # Standardize the target name to exactly 'Label'.
    if actual_target != TARGET_COLUMN:
        dataframe.rename(
            columns={actual_target: TARGET_COLUMN},
            inplace=True
        )

    # Remove rows with missing target labels.
    dataframe = dataframe.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    # Convert labels to strings for consistent encoding.
    dataframe[TARGET_COLUMN] = (
        dataframe[TARGET_COLUMN]
        .astype(str)
        .str.strip()
    )

    # Remove rows with empty labels.
    dataframe = dataframe[
        dataframe[TARGET_COLUMN] != ""
    ].copy()

    # Convert feature columns to numeric.
    feature_columns = [
        column for column in dataframe.columns
        if column != TARGET_COLUMN
    ]

    dataframe[feature_columns] = dataframe[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinite values with NaN.
    dataframe[feature_columns] = dataframe[
        feature_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # XGBoost can handle NaN feature values.
    # Missing values will be handled by XGBoost itself.

    print(
        f"{dataset_name} cleaned shape: {dataframe.shape}"
    )

    return dataframe


def get_available_features(ranked_features, train_df, test_df):
    """
    Keep ranked features that are present in both datasets.
    """
    available = []

    for feature in ranked_features:
        if (
            feature in train_df.columns
            and feature in test_df.columns
            and feature != TARGET_COLUMN
        ):
            available.append(feature)

    missing_train = [
        feature for feature in ranked_features
        if feature not in train_df.columns
    ]

    missing_test = [
        feature for feature in ranked_features
        if feature not in test_df.columns
    ]

    if missing_train:
        print(
            "\nWarning: Ranked features missing from training data:"
        )
        print(missing_train)

    if missing_test:
        print(
            "\nWarning: Ranked features missing from test data:"
        )
        print(missing_test)

    if not available:
        raise ValueError(
            "No ranked features were found in both datasets."
        )

    print(
        f"\nRanked features available in both datasets: "
        f"{len(available)}"
    )

    return available


# ============================================================
# 3. LOAD FEATURE RANKING
# ============================================================

print("=" * 65)
print("XGBOOST DDoS DETECTION")
print("Minimal Feature Selection and Computational Cost Analysis")
print("=" * 65)

ranked_features = load_feature_ranking(
    RANKING_FILE
)

print("\nFirst five ranked features:")
print(ranked_features[:5])


# ============================================================
# 4. LOAD TRAINING AND TEST DATASETS
# ============================================================

print("\nLoading training dataset...")

train_df = pd.read_csv(TRAIN_FILE)

print("Loading test dataset...")

test_df = pd.read_csv(TEST_FILE)

print("\nOriginal dataset shapes:")
print(f"Training dataset: {train_df.shape}")
print(f"Test dataset:     {test_df.shape}")


# ============================================================
# 5. CLEAN DATASETS
# ============================================================

print("\nCleaning datasets...")

train_df = clean_dataset(
    train_df, "Training dataset"
)

test_df = clean_dataset(
    test_df, "Test dataset"
)


# ============================================================
# 6. CHECK FEATURE AVAILABILITY
# ============================================================

available_features = get_available_features(
    ranked_features,
    train_df,
    test_df
)

# Use the same features in the same order for train and test.
max_features = min(
    max(FEATURE_SUBSETS),
    len(available_features)
)

if max_features < max(FEATURE_SUBSETS):
    print(
        "\nWarning: The number of available features is "
        "smaller than one or more requested subsets."
    )

print("\nFeature subset sizes to evaluate:")

effective_subsets = sorted(
    set(
        min(size, len(available_features))
        for size in FEATURE_SUBSETS
    )
)

print(effective_subsets)


# ============================================================
# 7. ENCODE TARGET LABELS
# ============================================================

print("\nEncoding target labels...")

label_encoder = LabelEncoder()

y_train_raw = train_df[TARGET_COLUMN]
y_test_raw = test_df[TARGET_COLUMN]

y_train = label_encoder.fit_transform(
    y_train_raw
)

# Reject unseen labels instead of silently encoding them
# incorrectly.
unseen_test_labels = sorted(
    set(y_test_raw.unique())
    - set(label_encoder.classes_)
)

if unseen_test_labels:
    raise ValueError(
        "The test dataset contains labels absent from "
        f"training data: {unseen_test_labels}"
    )

y_test = label_encoder.transform(
    y_test_raw
)

class_names = label_encoder.classes_

num_classes = len(class_names)

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print(f"\nNumber of classes: {num_classes}")

if num_classes < 2:
    raise ValueError(
        "At least two classes are required for classification."
    )

# Save label mapping for later reference.
label_mapping = {
    int(index): str(class_name)
    for index, class_name in enumerate(class_names)
}

with open(
    os.path.join(OUTPUT_DIR, "label_mapping.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        label_mapping,
        file,
        indent=4
    )


# ============================================================
# 8. PREPARE FEATURE SUBSETS
# ============================================================

print("\nPreparing feature subsets...")

for size in effective_subsets:
    print(
        f"Top-{size}: {available_features[:size]}"
    )


# ============================================================
# 9. TRAIN AND EVALUATE XGBOOST MODELS
# ============================================================

results = []

for subset_size in effective_subsets:

    print("\n" + "=" * 65)
    print(f"Training XGBoost with Top-{subset_size} features")
    print("=" * 65)

    selected_features = available_features[
        :subset_size
    ]

    print("\nSelected features:")
    print(selected_features)

    # Prepare train and test matrices.
    X_train = train_df[
        selected_features
    ].copy()

    X_test = test_df[
        selected_features
    ].copy()

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    # --------------------------------------------------------
    # TRAINING TIME
    # --------------------------------------------------------

    print("\nTraining model...")

    train_start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    train_end = time.perf_counter()

    training_time = train_end - train_start

    print(
        f"Training time: {training_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # MODEL SIZE
    # --------------------------------------------------------

    model_filename = (
        f"xgboost_top_{subset_size}_features.json"
    )

    model_path = os.path.join(
        OUTPUT_DIR,
        model_filename
    )

    model.save_model(model_path)

    model_size_bytes = os.path.getsize(
        model_path
    )

    model_size_mb = (
        model_size_bytes / (1024 * 1024)
    )

    print(
        f"Saved model: {model_path}"
    )

    print(
        f"Model size: {model_size_mb:.4f} MB"
    )

    # --------------------------------------------------------
    # PREDICTION TIME
    # --------------------------------------------------------

    print("\nPredicting test dataset...")

    prediction_start = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_end = time.perf_counter()

    prediction_time = (
        prediction_end - prediction_start
    )

    print(
        f"Prediction time: {prediction_time:.4f} seconds"
    )

    total_samples = len(y_test)

    if total_samples > 0:
        prediction_time_per_sample = (
            prediction_time / total_samples
        )

        samples_per_second = (
            total_samples / prediction_time
            if prediction_time > 0
            else 0
        )
    else:
        prediction_time_per_sample = 0
        samples_per_second = 0

    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nEvaluation results:")
    print(f"Accuracy:          {accuracy:.6f}")
    print(f"Macro Precision:   {precision_macro:.6f}")
    print(f"Macro Recall:      {recall_macro:.6f}")
    print(f"Macro F1-score:    {f1_macro:.6f}")
    print(f"Weighted Precision:{precision_weighted:.6f}")
    print(f"Weighted Recall:   {recall_weighted:.6f}")
    print(f"Weighted F1-score: {f1_weighted:.6f}")

    print(
        f"Prediction throughput: "
        f"{samples_per_second:.2f} samples/second"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        target_names=[
            str(name) for name in class_names
        ],
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_filename = (
        f"classification_report_top_{subset_size}.csv"
    )

    report_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            report_filename
        ),
        index=True
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(num_classes)
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{name}" for name in class_names
        ],
        columns=[
            f"Predicted_{name}" for name in class_names
        ]
    )

    confusion_filename = (
        f"confusion_matrix_top_{subset_size}.csv"
    )

    cm_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            confusion_filename
        ),
        index=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_filename = (
        f"feature_importance_top_{subset_size}.csv"
    )

    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            importance_filename
        ),
        index=False
    )

    # --------------------------------------------------------
    # SAVE SUMMARY RESULTS
    # --------------------------------------------------------

    result = {
        "Algorithm": "XGBoost",
        "Feature_Count": subset_size,
        "Selected_Features": ", ".join(
            selected_features
        ),
        "Accuracy": accuracy,
        "Precision_Macro": precision_macro,
        "Recall_Macro": recall_macro,
        "F1_Macro": f1_macro,
        "Precision_Weighted": precision_weighted,
        "Recall_Weighted": recall_weighted,
        "F1_Weighted": f1_weighted,
        "Training_Time_Seconds": training_time,
        "Prediction_Time_Seconds": prediction_time,
        "Prediction_Time_Per_Sample_Seconds":
            prediction_time_per_sample,
        "Prediction_Throughput_Samples_Per_Second":
            samples_per_second,
        "Model_Size_MB": model_size_mb,
        "Training_Samples": len(y_train),
        "Test_Samples": total_samples,
        "Number_Of_Classes": num_classes,
        "N_Estimators": N_ESTIMATORS,
        "Max_Depth": MAX_DEPTH,
        "Learning_Rate": LEARNING_RATE
    }

    results.append(result)

    print(
        f"\nCompleted Top-{subset_size} feature experiment."
    )


# ============================================================
# 10. SAVE FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("SAVING FINAL RESULTS")
print("=" * 65)

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

print(
    f"\nResults saved to:\n{RESULTS_FILE}"
)

print(
    f"\nModels and reports saved in:\n{OUTPUT_DIR}"
)


# ============================================================
# 11. DISPLAY COMPARISON TABLE
# ============================================================

print("\n" + "=" * 65)
print("XGBOOST FEATURE SUBSET COMPARISON")
print("=" * 65)

display_columns = [
    "Feature_Count",
    "Accuracy",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "Training_Time_Seconds",
    "Prediction_Time_Seconds",
    "Model_Size_MB"
]

print(
    results_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ============================================================
# 12. IDENTIFY BEST RESULTS
# ============================================================

best_f1_row = results_df.loc[
    results_df["F1_Macro"].idxmax()
]

fastest_row = results_df.loc[
    results_df["Training_Time_Seconds"].idxmin()
]

smallest_model_row = results_df.loc[
    results_df["Model_Size_MB"].idxmin()
]

print("\nBest Macro F1-score:")
print(
    f"Feature count: {best_f1_row['Feature_Count']}"
)
print(
    f"Macro F1-score: {best_f1_row['F1_Macro']:.6f}"
)

print("\nFastest training:")
print(
    f"Feature count: {fastest_row['Feature_Count']}"
)
print(
    f"Training time: "
    f"{fastest_row['Training_Time_Seconds']:.4f} seconds"
)

print("\nSmallest saved model:")
print(
    f"Feature count: {smallest_model_row['Feature_Count']}"
)
print(
    f"Model size: {smallest_model_row['Model_Size_MB']:.4f} MB"
)


# ============================================================
# 13. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("XGBOOST EXPERIMENTS COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated files:")
print(f"1. Results CSV: {RESULTS_FILE}")
print(f"2. Output directory: {OUTPUT_DIR}")
print("3. Trained models for each feature subset")
print("4. Classification reports")
print("5. Confusion matrices")
print("6. Feature importance reports")
print("7. Label mapping JSON")