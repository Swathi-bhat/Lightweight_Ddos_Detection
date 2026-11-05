import os
import time
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# ==============================
# CONFIGURATION
# ==============================

TRAIN_PATH = "data/train_dataset.csv"
OUTPUT_PATH = "feature_selection/information_gain_ranking.csv"

TARGET_COLUMN = "Label"
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

start_time = time.time()

# ==============================
# LOAD DATASET
# ==============================

print("Loading training dataset...", flush=True)

train_data = pd.read_csv(TRAIN_PATH)

print(f"Dataset shape: {train_data.shape}", flush=True)

if TARGET_COLUMN not in train_data.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found.\n"
        f"Available columns: {train_data.columns.tolist()}"
    )

# ==============================
# SAMPLE DATA
# ==============================

if len(train_data) > SAMPLE_SIZE:
    print(f"Sampling {SAMPLE_SIZE:,} rows...", flush=True)

    sampled_data = train_data.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )
else:
    sampled_data = train_data

# ==============================
# SEPARATE FEATURES AND LABEL
# ==============================

X = sampled_data.drop(columns=[TARGET_COLUMN])
y = sampled_data[TARGET_COLUMN]

# Keep numeric features only
X = X.select_dtypes(include="number")

if X.shape[1] == 0:
    raise ValueError("No numeric features found in the dataset.")

# Replace infinite values and handle missing values
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median()).fillna(0)

# ==============================
# FEATURE SELECTION
# ==============================

print(
    f"Calculating Mutual Information for {X.shape[1]} features...",
    flush=True
)

scores = mutual_info_classif(
    X,
    y,
    random_state=RANDOM_STATE,
    n_neighbors=3
)

# ==============================
# RANK FEATURES
# ==============================

ranking = pd.DataFrame({
    "Feature": X.columns,
    "Information_Gain": scores
})

ranking = ranking.sort_values(
    by="Information_Gain",
    ascending=False
).reset_index(drop=True)

ranking.insert(
    0,
    "Rank",
    range(1, len(ranking) + 1)
)

# ==============================
# SAVE RESULTS
# ==============================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

ranking.to_csv(OUTPUT_PATH, index=False)

print("\n========== FEATURE RANKING ==========", flush=True)
print(ranking.head(10).to_string(index=False), flush=True)

print(
    f"\nComplete ranking saved to: {OUTPUT_PATH}",
    flush=True
)

print(
    f"Total execution time: {time.time() - start_time:.2f} seconds",
    flush=True
)


scores = mutual_info_classif(
    X,
    y,
    random_state=RANDOM_STATE,
    n_neighbors=3
)

ranking = pd.DataFrame({
    "Feature": X.columns,
    "Information_Gain": scores
})

ranking = ranking.sort_values(
    by="Information_Gain",
    ascending=False
).reset_index(drop=True)

ranking.insert(0, "Rank", range(1, len(ranking) + 1))

ranking.to_csv(OUTPUT_PATH, index=False)
