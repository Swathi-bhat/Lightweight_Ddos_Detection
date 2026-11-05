from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "saved_models" / "best_model.joblib"
METADATA_FILE = PROJECT_ROOT / "saved_models" / "model_metadata.json"

# Corrected path: comparison results are stored in results/
COMPARISON_FILE = (
    PROJECT_ROOT / "results" / "model_comparison_results.csv"
)


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Lightweight DDoS Detection",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# 3. LOAD MODEL AND METADATA
# ==================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


# ==================================================
# 4. CHECK REQUIRED FILES
# ==================================================

st.sidebar.title("🛡️ DDoS Detection")
st.sidebar.caption("Machine Learning-Based Traffic Analysis")

missing_files = []

if not MODEL_FILE.exists():
    missing_files.append(str(MODEL_FILE))

if not METADATA_FILE.exists():
    missing_files.append(str(METADATA_FILE))

if missing_files:
    st.error("Required model files were not found.")

    for file_path in missing_files:
        st.code(file_path)

    st.info(
        "Check that best_model.joblib and model_metadata.json "
        "exist inside the saved_models folder. "
        "You do not need to rerun model comparison if these "
        "files have already been generated."
    )

    st.stop()


# Load the saved model
try:
    model, metadata = load_model()

except Exception as error:
    st.error("Could not load the saved model.")
    st.exception(error)
    st.stop()


# Read metadata
try:
    best_model_name = metadata["model_name"]
    best_accuracy = float(metadata["accuracy"])
    selected_features = metadata["selected_features"]
    label_classes = metadata["classes"]

except (KeyError, TypeError, ValueError) as error:
    st.error("Could not read the required model metadata.")
    st.exception(error)
    st.stop()


# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Model Comparison",
        "DDoS Detection",
        "Computational Cost",
       
    ]
)


# ==================================================
# 6. OVERVIEW PAGE
# ==================================================

if page == "Overview":

    st.title("🛡️ Lightweight DDoS Detection System")

    st.markdown(
        """
        ## Project Overview

        **Lightweight DDoS Detection Using Minimal Feature
        Selection and Computational Cost Analysis**

        This application uses machine learning to classify
        network traffic using selected traffic features.

        ### Project Workflow

        1. Rank features using Information Gain.
        2. Select the most informative features.
        3. Train and compare machine-learning models.
        4. Evaluate model performance.
        5. Measure training and prediction time.
        6. Use the selected trained model to classify traffic.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Selected Model",
        str(best_model_name)
    )

    col2.metric(
        "Test Accuracy",
        f"{best_accuracy * 100:.2f}%"
    )

    col3.metric(
        "Selected Features",
        len(selected_features)
    )

    st.subheader("Selected Features")

    st.write(selected_features)

    st.info(
        "Upload a network traffic CSV file on the DDoS Detection "
        "page to classify its records. This application does not "
        "capture live network packets."
    )

    st.warning(
        "Predictions are model outputs and are not a guarantee "
        "that traffic is malicious or safe."
    )


# ==================================================
# 7. MODEL COMPARISON PAGE
# ==================================================

elif page == "Model Comparison":

    st.title("📊 Model Comparison")

    if not COMPARISON_FILE.exists():

        st.error(
            f"Comparison results were not found at:\n"
            f"{COMPARISON_FILE}"
        )

        st.info(
            "Check the CSV filename in your results folder. "
            "You do not need to rerun model comparison just "
            "to correct this path."
        )

        st.stop()

    try:
        comparison_df = pd.read_csv(COMPARISON_FILE)

    except Exception as error:
        st.error("Could not read the comparison CSV.")
        st.exception(error)
        st.stop()

    if comparison_df.empty:
        st.warning("The comparison CSV contains no results.")
        st.stop()

    st.subheader("Model Performance Results")

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # Find columns using common naming conventions
    accuracy_col = next(
        (
            col for col in comparison_df.columns
            if col.lower() == "accuracy"
        ),
        None
    )

    model_col = next(
        (
            col for col in comparison_df.columns
            if col.lower() == "model"
        ),
        None
    )

    if accuracy_col and model_col:

        st.subheader("Accuracy by Model")

        chart_df = comparison_df[[model_col, accuracy_col]].copy()

        chart_df[accuracy_col] = pd.to_numeric(
            chart_df[accuracy_col],
            errors="coerce"
        )

        chart_df = chart_df.dropna(subset=[accuracy_col])

        if not chart_df.empty:

            # Average accuracy if the CSV has multiple rows
            # for each model using different feature counts.
            chart_df = (
                chart_df.groupby(model_col)[accuracy_col]
                .mean()
                .sort_values(ascending=False)
            )

            st.bar_chart(chart_df)

    st.success(
        f"Selected model: {best_model_name} | "
        f"Test accuracy: {best_accuracy * 100:.2f}%"
    )

    st.download_button(
        label="Download Comparison Results",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )


# ==================================================
# 8. DDOS DETECTION PAGE
# ==================================================

elif page == "DDoS Detection":

    st.title("🔍 DDoS Traffic Detection")

    st.write(
        "Upload a CSV file containing network traffic records. "
        "The saved model will predict the class of each record."
    )

    st.markdown("### Required Features")

    with st.expander("View required feature columns"):

        st.write(selected_features)

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"]
    )
    


#############################################################################33

    if uploaded_file is not None:

        try:
            input_df = pd.read_csv(
                uploaded_file,
                low_memory=False
            )

            # Remove accidental spaces from column names
            input_df.columns = (
                input_df.columns.astype(str).str.strip()
            )

            st.subheader("Uploaded Data")

            st.dataframe(
                input_df.head(10),
                use_container_width=True,
                hide_index=True
            )

            if input_df.empty:

                st.error("The uploaded CSV contains no rows.")

            else:

                # Check for missing required features
                missing_features = [
                    feature
                    for feature in selected_features
                    if feature not in input_df.columns
                ]

                if missing_features:

                    st.error(
                        "The uploaded CSV is missing these required "
                        "features:\n\n"
                        + ", ".join(missing_features)
                    )

                else:

                    st.info(
                        f"Records uploaded: {len(input_df)}"
                    )

                    if st.button(
                        "Run DDoS Detection",
                        type="primary"
                    ):

                        # Keep the same feature order as training
                        X_input = input_df[
                            selected_features
                        ].copy()

                        # Convert feature values to numeric
                        for col in selected_features:

                            X_input[col] = pd.to_numeric(
                                X_input[col],
                                errors="coerce"
                            )

                        # Replace infinite values with NaN
                        X_input = X_input.replace(
                            [np.inf, -np.inf],
                            np.nan
                        )

                        # Make predictions
                        predictions = model.predict(X_input)

                        # Convert encoded predictions to labels
                        # when the model outputs numeric class IDs.
                        classes = [
                            str(label)
                            for label in label_classes
                        ]

                        predicted_labels = []

                        for prediction in predictions:

                            if isinstance(
                                prediction,
                                (int, np.integer)
                            ) or (
                                isinstance(
                                    prediction,
                                    (float, np.floating)
                                )
                                and float(prediction).is_integer()
                            ):

                                class_index = int(prediction)

                                if 0 <= class_index < len(classes):
                                    predicted_labels.append(
                                        classes[class_index]
                                    )
                                else:
                                    predicted_labels.append(
                                        str(prediction)
                                    )

                            else:
                                predicted_labels.append(
                                    str(prediction)
                                )

                        # Add predictions to the uploaded data
                        results_df = input_df.copy()

                        results_df["Predicted_Label"] = (
                            predicted_labels
                        )

                        st.subheader("Detection Results")

                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Summarize predicted classes
                        st.subheader("Prediction Summary")

                        summary = (
                            results_df["Predicted_Label"]
                            .value_counts()
                            .rename_axis("Predicted Class")
                            .reset_index(name="Number of Rows")
                        )

                        st.dataframe(
                            summary,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Plot prediction counts
                        st.subheader("Traffic Classification")

                        st.bar_chart(
                            summary.set_index("Predicted Class")
                            [["Number of Rows"]]
                        )

                        # Download results
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=results_df.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="ddos_detection_results.csv",
                            mime="text/csv"
                        )

                        st.success(
                            f"Successfully processed "
                            f"{len(results_df)} records."
                        )

                        st.caption(
                            "The predicted classes depend on the "
                            "labels and preprocessing used when "
                            "the model was trained."
                        )

        except Exception as error:

            st.error(
                "Could not process this CSV. Check the input "
                "format, feature columns, and model preprocessing."
            )

            st.exception(error)


# ==================================================
# 9. COMPUTATIONAL COST PAGE
# ==================================================

elif page == "Computational Cost":

    st.title("⚙️ Computational Cost Analysis")

    if not COMPARISON_FILE.exists():

        st.error(
            f"Comparison results were not found at:\n"
            f"{COMPARISON_FILE}"
        )

        st.stop()

    try:
        comparison_df = pd.read_csv(COMPARISON_FILE)

    except Exception as error:
        st.error("Could not read the comparison CSV.")
        st.exception(error)
        st.stop()

    st.subheader("Computational Cost Results")

    # st.dataframe(
    #     comparison_df,
    #     use_container_width=True,
    #     hide_index=True
    # )

    # Locate timing columns regardless of minor naming differences
    time_columns = {
        "Training Time": [
            "Training_Time_Seconds",
            "Training_Time_s",
            "Training_Time"
        ],
        "Prediction Time": [
            "Prediction_Time_Seconds",
            "Prediction_Time_s",
            "Prediction_Time"
        ],
        "Prediction Latency": [
            "Prediction_Time_ms_per_row",
            "Prediction_Time_per_Sample_us"
        ]
    }

    model_col = next(
        (
            col for col in comparison_df.columns
            if col.lower() == "model"
        ),
        None
    )

    if model_col:

        for title, possible_columns in time_columns.items():

            time_col = next(
                (
                    col for col in possible_columns
                    if col in comparison_df.columns
                ),
                None
            )

            if time_col:

                st.subheader(title)

                chart_df = comparison_df[
                    [model_col, time_col]
                ].copy()

                chart_df[time_col] = pd.to_numeric(
                    chart_df[time_col],
                    errors="coerce"
                )

                chart_df = chart_df.dropna(
                    subset=[time_col]
                )

                if not chart_df.empty:

                    st.bar_chart(
                        chart_df.set_index(model_col)[[time_col]]
                    )

                else:
                    st.info(
                        f"No numeric values available for {title}."
                    )

    st.caption(
        "Training and prediction times depend on hardware, "
        "dataset size, software versions, and system load."
    )

# ==================================================
# 11. FOOTER
# ==================================================

st.sidebar.markdown("---")
st.sidebar.caption(
    "Lightweight DDoS Detection | Machine Learning Project"
)