import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier


LABEL = "Label"
FONT = "Arial"
NUMBER_OF_FEATURES = 5
RANDOM_STATE = 42


def get_project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_selected_features(project_root, train, test):
    """Load the top five features from the Information Gain ranking."""

    ranking_path = (
        project_root
        / "feature_selection"
        / "information_gain_ranking.csv"
    )

    if not ranking_path.exists():
        raise FileNotFoundError(
            f"Feature ranking file not found: {ranking_path}"
        )

    ranking = pd.read_csv(ranking_path)

    if ranking.empty or len(ranking.columns) == 0:
        raise ValueError("The feature ranking CSV is empty.")

    # Identify the column containing feature names.
    possible_names = {"feature", "features", "feature_name"}

    feature_column = next(
        (
            column
            for column in ranking.columns
            if column.strip().lower() in possible_names
        ),
        ranking.columns[0],
    )

    selected_features = (
        ranking[feature_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    # Avoid duplicate feature names.
    selected_features = list(dict.fromkeys(selected_features))

    if len(selected_features) < NUMBER_OF_FEATURES:
        raise ValueError(
            f"At least {NUMBER_OF_FEATURES} ranked features are required. "
            f"Found {len(selected_features)}."
        )

    selected_features = selected_features[:NUMBER_OF_FEATURES]

    missing_train = [
        feature for feature in selected_features
        if feature not in train.columns
    ]

    missing_test = [
        feature for feature in selected_features
        if feature not in test.columns
    ]

    if missing_train or missing_test:
        raise ValueError(
            "Some selected features are missing from the datasets.\n"
            f"Missing from training data: {missing_train}\n"
            f"Missing from testing data: {missing_test}"
        )

    return selected_features


def train_and_evaluate(train_path, test_path):
    """Train, evaluate, and save the selected Decision Tree."""

    project_root = get_project_root()

    train_path = Path(train_path)
    test_path = Path(test_path)

    # Resolve relative dataset paths from the project root.
    if not train_path.is_absolute():
        train_path = project_root / train_path

    if not test_path.is_absolute():
        test_path = project_root / test_path

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {test_path}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Training dataset shape: {train.shape}")
    print(f"Testing dataset shape: {test.shape}")

    if LABEL not in train.columns or LABEL not in test.columns:
        raise ValueError(
            f"Both datasets must contain the target column '{LABEL}'."
        )

    # Select the five ranked features.
    selected_features = load_selected_features(
        project_root, train, test
    )

    print("\nSelected features:")
    for index, feature in enumerate(selected_features, start=1):
        print(f"{index}. {feature}")

    X_train = train[selected_features].copy()
    y_train = train[LABEL]

    X_test = test[selected_features].copy()
    y_test = test[LABEL]

    # Convert selected features to numeric values.
    for feature in selected_features:
        X_train[feature] = pd.to_numeric(
            X_train[feature], errors="coerce"
        )
        X_test[feature] = pd.to_numeric(
            X_test[feature], errors="coerce"
        )

    # Replace infinite values and handle missing values consistently.
    X_train = X_train.replace([float("inf"), float("-inf")], float("nan"))
    X_test = X_test.replace([float("inf"), float("-inf")], float("nan"))

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Train the Decision Tree.
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE)

    start_time = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    # Measure prediction time.
    start_time = time.perf_counter()
    y_pred = clf.predict(X_test)
    predict_time = time.perf_counter() - start_time

    # Calculate overall performance metrics.
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
    }

    for average in ("weighted", "macro"):
        metrics[f"Precision ({average})"] = precision_score(
            y_test,
            y_pred,
            average=average,
            zero_division=0,
        )

        metrics[f"Recall ({average})"] = recall_score(
            y_test,
            y_pred,
            average=average,
            zero_division=0,
        )

        metrics[f"F1-score ({average})"] = f1_score(
            y_test,
            y_pred,
            average=average,
            zero_division=0,
        )

    # Per-class metrics.
    classes = list(clf.classes_)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=classes,
            zero_division=0,
        )
    )

    per_class = pd.DataFrame(
        {
            "Class": classes,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Support": support,
        }
    )

    # Confusion matrix.
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=classes),
        index=[f"Actual: {label}" for label in classes],
        columns=[f"Predicted: {label}" for label in classes],
    )

    # Feature importance for the five selected features.
    fi = (
        pd.DataFrame(
            {
                "Feature": selected_features,
                "Importance": clf.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    prediction_time_per_sample_us = (
        predict_time / len(X_test) * 1e6
        if len(X_test) > 0
        else 0.0
    )

    info = {
        "Training samples": len(train),
        "Test samples": len(test),
        "Number of features": len(selected_features),
        "Number of classes": len(classes),
        "Tree depth": clf.get_depth(),
        "Number of leaves": clf.get_n_leaves(),
        "Training time (s)": train_time,
        "Prediction time (s)": predict_time,
        "Prediction time per sample (µs)": (
            prediction_time_per_sample_us
        ),
    }

    # Save the final fitted model and metadata.
    saved_models_dir = project_root / "saved_models"
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    model_path = saved_models_dir / "best_model.joblib"
    metadata_path = saved_models_dir / "model_metadata.json"

    # Save the estimator directly so the existing app.py can load it
    # with joblib.load(MODEL_FILE).
    joblib.dump(clf, model_path)

    metadata = {
        "model_name": "Decision Tree",
        "target_column": LABEL,
        "selected_features": selected_features,
        "classes": [str(label) for label in classes],
        "accuracy": float(metrics["Accuracy"]),
        "precision_weighted": float(metrics["Precision (weighted)"]),
        "recall_weighted": float(metrics["Recall (weighted)"]),
        "f1_weighted": float(metrics["F1-score (weighted)"]),
        "number_of_features": len(selected_features),
        "training_time_seconds": float(train_time),
        "prediction_time_seconds": float(predict_time),
        "prediction_time_per_sample_us": float(
            prediction_time_per_sample_us
        ),
        "random_state": RANDOM_STATE,
        "missing_value_fill": 0,
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nFinal model saved successfully.")
    print(f"Model path: {model_path}")
    print(f"Metadata path: {metadata_path}")

    return metrics, info, per_class, cm, fi


def style_sheet(ws, pct_cols=(), num_formats=None, header_row=1):
    """Apply consistent Excel formatting."""

    header_fill = PatternFill(
        "solid",
        start_color="1F4E78",
    )

    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name=FONT, size=10)

    for cell in ws[header_row]:
        cell.font = Font(
            name=FONT,
            size=10,
            bold=True,
            color="FFFFFF",
        )
        cell.fill = header_fill
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

    for column in ws.columns:
        width = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in column
        )

        ws.column_dimensions[
            get_column_letter(column[0].column)
        ].width = min(max(width + 3, 12), 45)

    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)


def write_excel(out_path, metrics, info, per_class, cm, fi):
    """Write evaluation results to an Excel workbook."""

    out_path = Path(out_path)

    if not out_path.is_absolute():
        out_path = get_project_root() / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # Summary sheet.
        rows = [("Performance Metric", "Value")]
        rows += [(key, value) for key, value in metrics.items()]
        rows += [("", ""), ("Timing & Model Info", "Value")]
        rows += [(key, value) for key, value in info.items()]

        pd.DataFrame(rows).to_excel(
            writer,
            sheet_name="Summary",
            index=False,
            header=False,
        )

        per_class.to_excel(
            writer,
            sheet_name="Per-Class Metrics",
            index=False,
        )

        cm.to_excel(
            writer,
            sheet_name="Confusion Matrix",
        )

        fi.to_excel(
            writer,
            sheet_name="Feature Importance",
            index=False,
        )

        # Format summary sheet.
        ws = writer.sheets["Summary"]
        style_sheet(ws)

        for row_index in range(1, ws.max_row + 1):
            label_cell = ws.cell(row_index, 1)
            value_cell = ws.cell(row_index, 2)

            if label_cell.value == "Timing & Model Info":
                for cell in (label_cell, value_cell):
                    cell.font = Font(
                        name=FONT,
                        size=10,
                        bold=True,
                        color="FFFFFF",
                    )
                    cell.fill = PatternFill(
                        "solid",
                        start_color="1F4E78",
                    )
                    cell.alignment = Alignment(
                        horizontal="center"
                    )

            if label_cell.value in metrics:
                value_cell.number_format = "0.0000%"

            elif label_cell.value in (
                "Training time (s)",
                "Prediction time (s)",
            ):
                value_cell.number_format = "0.000000"

            elif label_cell.value == (
                "Prediction time per sample (µs)"
            ):
                value_cell.number_format = "0.000000"

            elif label_cell.value in info:
                value_cell.number_format = "#,##0"

            value_cell.alignment = Alignment(
                horizontal="right"
            )

        # Format per-class metrics.
        ws = writer.sheets["Per-Class Metrics"]
        style_sheet(ws)

        for row_index in range(2, ws.max_row + 1):
            for column_index in (2, 3, 4):
                ws.cell(
                    row_index, column_index
                ).number_format = "0.0000"

            ws.cell(row_index, 5).number_format = "#,##0"

        # Format confusion matrix.
        ws = writer.sheets["Confusion Matrix"]
        style_sheet(ws)

        for row_index in range(2, ws.max_row + 1):
            ws.cell(row_index, 1).font = Font(
                name=FONT,
                size=10,
                bold=True,
            )

            for column_index in range(2, ws.max_column + 1):
                ws.cell(
                    row_index, column_index
                ).number_format = "#,##0"

        # Format feature importance.
        ws = writer.sheets["Feature Importance"]
        style_sheet(ws)

        for row_index in range(2, ws.max_row + 1):
            ws.cell(row_index, 2).number_format = "0.0000"

    print(f"\nExcel results written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default="data/train_dataset.csv",
        help="Path to the training CSV.",
    )

    parser.add_argument(
        "--test",
        default="data/test_dataset.csv",
        help="Path to the testing CSV.",
    )

    parser.add_argument(
        "--out",
        default="decision_tree_results.xlsx",
        help="Output Excel workbook path.",
    )

    args = parser.parse_args()

    results = train_and_evaluate(
        args.train,
        args.test,
    )

    metrics, info, per_class, cm, fi = results

    print("\n========== MODEL PERFORMANCE ==========")

    for key, value in {**metrics, **info}.items():
        print(f"{key}: {value}")

    write_excel(
        args.out,
        metrics,
        info,
        per_class,
        cm,
        fi,
    )
