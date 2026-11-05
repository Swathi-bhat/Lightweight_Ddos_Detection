import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "results" / "model_comparison_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "results" / "model_selection_accuracy_time.csv"
)

# Selection weights
ACCURACY_WEIGHT = 0.70
SPEED_WEIGHT = 0.30

# Load existing results; no training is performed
df = pd.read_csv(INPUT_FILE)

# Use measured per-sample prediction latency
df["Accuracy_Score"] = df["Accuracy"] / df["Accuracy"].max()

df["Speed_Score"] = (
    df["Prediction_Time_per_Sample_us"].min()
    / df["Prediction_Time_per_Sample_us"]
)

# Calculate combined score
df["Combined_Score"] = (
    ACCURACY_WEIGHT * df["Accuracy_Score"]
    + SPEED_WEIGHT * df["Speed_Score"]
)

# Rank all model and feature-count combinations
df = df.sort_values(
    "Combined_Score",
    ascending=False
).reset_index(drop=True)

df["Rank"] = df.index + 1

# Save the ranking
df.to_csv(OUTPUT_FILE, index=False)

# Display the top five
columns = [
    "Rank",
    "Model",
    "Number_of_Features",
    "Accuracy",
    "Prediction_Time_per_Sample_us",
    "Combined_Score"
]

print("\nTOP 5 MODEL CONFIGURATIONS\n")
print(df[columns].head(5).to_string(index=False))

print("\nSelected configuration:")
print(f"Model: {df.iloc[0]['Model']}")
print(f"Features: {df.iloc[0]['Number_of_Features']}")
print(f"Accuracy: {df.iloc[0]['Accuracy']:.6%}")
print(
    "Prediction time per sample: "
    f"{df.iloc[0]['Prediction_Time_per_Sample_us']:.6f} us"
)
print(f"Combined score: {df.iloc[0]['Combined_Score']:.6f}")

print(f"\nSaved ranking to: {OUTPUT_FILE}")