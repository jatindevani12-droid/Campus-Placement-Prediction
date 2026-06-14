"""
CAMPUS PLACEMENT - Steps 1, 2 & 3
  Step 1: Load & explore data
  Step 2: Preprocess (encode, scale, handle NaN)
  Step 3: Exploratory Data Analysis summary
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------------------------------
# STEP 1 - LOAD & EXPLORE DATA
# ------------------------------------------------
print("=" * 60)
print("STEP 1 - DATA LOADING & EXPLORATION")
print("=" * 60)

df = pd.read_csv("placement.csv")

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print("--- First 5 rows ---")
print(df.head())

print("\n--- Column Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Basic Statistics ---")
print(df.describe())

# ------------------------------------------------
# STEP 2 - PREPROCESSING
# ------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 - DATA PREPROCESSING")
print("=" * 60)

# Drop serial number column
df.drop(columns=["sl_no"], inplace=True)

# Fill missing salary with 0 (only Not Placed students are NaN)
df["salary"] = df["salary"].fillna(0)

# Binary encoding
df["gender"]  = df["gender"].map({"M": 0, "F": 1})
df["workex"]  = df["workex"].map({"No": 0, "Yes": 1})
df["status"]  = df["status"].map({"Not Placed": 0, "Placed": 1})

# Label encoding for multi-class categoricals
le = LabelEncoder()
for col in ["ssc_b", "hsc_b", "hsc_s", "degree_t", "specialisation"]:
    df[col] = le.fit_transform(df[col])

print("\nCategorical encoding complete.")
print("\n--- Processed head ---")
print(df.head())

print("\n--- Placed vs Not Placed counts ---")
print(df["status"].value_counts().rename({1: "Placed", 0: "Not Placed"}))

# Feature matrix and target
FEATURES = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p",
            "gender", "workex", "specialisation", "hsc_s", "degree_t"]

X = df[FEATURES]
y = df["status"]

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling complete.")
print(f"   Feature matrix shape: {X_scaled.shape}")

# ------------------------------------------------
# STEP 3 - EDA SUMMARY (printed stats)
# ------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

total      = len(df)
placed     = int(df["status"].sum())
not_placed = total - placed

print(f"\nOverall Placement Rate: {placed}/{total} = {placed/total*100:.1f}%")

# Placement by gender
print("\n--- Placement by Gender ---")
for g_code, g_name in {0: "Male", 1: "Female"}.items():
    subset = df[df["gender"] == g_code]
    rate   = subset["status"].mean() * 100
    print(f"  {g_name}: {rate:.1f}% placed  ({len(subset)} students)")

# Placement by work experience
print("\n--- Placement by Work Experience ---")
for w_code, w_name in {0: "No WorkEx", 1: "Has WorkEx"}.items():
    subset = df[df["workex"] == w_code]
    rate   = subset["status"].mean() * 100
    print(f"  {w_name}: {rate:.1f}% placed  ({len(subset)} students)")

# Avg scores: placed vs not placed
print("\n--- Average Scores: Placed vs Not Placed ---")
stats = df.groupby("status")[["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]].mean()
stats.index = stats.index.map({0: "Not Placed", 1: "Placed"})
print(stats.round(2))

# Avg salary for placed students
placed_df = df[df["status"] == 1]
print(f"\n--- Salary Stats (Placed Only, n={len(placed_df)}) ---")
print(f"  Mean   : {placed_df['salary'].mean():,.0f}")
print(f"  Median : {placed_df['salary'].median():,.0f}")
print(f"  Min    : {placed_df['salary'].min():,.0f}")
print(f"  Max    : {placed_df['salary'].max():,.0f}")

print("\nEDA complete. Run visualization.py for charts.")
print("Run ml_models.py to train all ML models.")