"""
CAMPUS PLACEMENT - Step 5: Train 4 ML Models
Models: Logistic Regression, Decision Tree, Random Forest, KNN
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report)

print("=" * 65)
print("STEP 5 - TRAINING 4 ML MODELS")
print("=" * 65)

# ---------------------------------
# LOAD & PREPROCESS
# ---------------------------------
df = pd.read_csv("placement.csv")
df.drop(columns=["sl_no"], inplace=True)
df["salary"] = df["salary"].fillna(0)

le = LabelEncoder()
for col in ["ssc_b", "hsc_b", "hsc_s", "degree_t", "specialisation"]:
    df[col] = le.fit_transform(df[col])

df["gender"] = df["gender"].map({"M": 0, "F": 1})
df["workex"] = df["workex"].map({"No": 0, "Yes": 1})
df["status"] = df["status"].map({"Not Placed": 0, "Placed": 1})

FEATURES = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p",
            "gender", "workex", "specialisation", "hsc_s", "degree_t"]

X = df[FEATURES].values
y = df["status"].values

# ---------------------------------
# TRAIN / TEST SPLIT (stratified)
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------
# SCALING
# ---------------------------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ---------------------------------
# MODEL DEFINITIONS
# ---------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
}

# ---------------------------------
# TRAIN & EVALUATE EACH MODEL
# ---------------------------------
results = {}

print(f"\n{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'CV Acc':>8}")
print("-" * 65)

best_f1 = 0
best_model_name = ""

for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    y_pred    = clf.predict(X_test_sc)
    y_prob    = clf.predict_proba(X_test_sc)[:, 1]
    cv_scores = cross_val_score(clf, X_train_sc, y_train, cv=5, scoring="accuracy")

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc_s  = roc_auc_score(y_test, y_prob)
    cv   = cv_scores.mean()

    results[name] = {
        "model":     clf,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "roc_auc":   auc_s,
        "cv_acc":    cv,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }

    print(f"{name:<22} {acc:>7.3f} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {auc_s:>7.3f} {cv:>8.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

# ---------------------------------
# DETAILED REPORT FOR BEST MODEL
# ---------------------------------
print(f"\nBest Model (by F1): {best_model_name}")
print("-" * 40)
best = results[best_model_name]
print(classification_report(y_test, best["y_pred"],
                            target_names=["Not Placed", "Placed"]))

# ---------------------------------
# FEATURE IMPORTANCE (Random Forest)
# ---------------------------------
rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n--- Random Forest Feature Importances ---")
for _, row in feat_df.iterrows():
    bar = "#" * int(row["Importance"] * 60)
    print(f"  {row['Feature']:<18} {row['Importance']:.4f}  {bar}")

# ---------------------------------
# SAVE ARTIFACTS
# ---------------------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(results[best_model_name]["model"], f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save results summary as CSV
summary_rows = []
for name, r in results.items():
    summary_rows.append({
        "Model":     name,
        "Accuracy":  round(r["accuracy"],  4),
        "Precision": round(r["precision"], 4),
        "Recall":    round(r["recall"],    4),
        "F1":        round(r["f1"],        4),
        "ROC_AUC":   round(r["roc_auc"],   4),
        "CV_Acc":    round(r["cv_acc"],    4),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("model_results.csv", index=False)

print(f"\nBest model saved -> best_model.pkl")
print(f"Scaler saved     -> scaler.pkl")
print(f"Results saved    -> model_results.csv")
print(f"Run evaluate.py for charts. Run app.py for dashboard.")
