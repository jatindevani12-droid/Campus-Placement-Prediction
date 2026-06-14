"""
CAMPUS PLACEMENT - Step 6: Model Evaluation & Comparison Charts
Run AFTER ml_models.py (requires model_results.csv)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    classification_report
)

# ── Style ───────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3a",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#444",
})

COLORS = ["#6C63FF", "#FF6584", "#43C59E", "#FF9F43", "#54A0FF"]
MODEL_COLORS = {
    "Logistic Regression": "#6C63FF",
    "Decision Tree":       "#FF9F43",
    "Random Forest":       "#43C59E",
    "KNN":                 "#FF6584",
}

# ──────────────────────────────
# Rebuild data + models (needed for ROC/CM)
# ──────────────────────────────
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
}

trained = {}
for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    trained[name] = clf

# ─────────────────────────────────────────────────────
# EVAL CHART 1 — Model Accuracy Comparison (grouped metric bar)
# ─────────────────────────────────────────────────────
metrics_data = {}
for name, clf in trained.items():
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score)
    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]
    metrics_data[name] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
    }

metrics_df = pd.DataFrame(metrics_data).T
model_names = list(metrics_df.index)
metric_cols = list(metrics_df.columns)

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle("Eval Chart 1 — Model Performance Comparison", fontsize=16,
             fontweight="bold", color="#e0e0e0")

x = np.arange(len(model_names))
width = 0.15
for i, metric in enumerate(metric_cols):
    offset = (i - 2) * width
    bars = ax.bar(x + offset, metrics_df[metric], width,
                  label=metric, color=COLORS[i], edgecolor="#0f1117",
                  alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#e0e0e0")

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=20, ha="right")
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("eval1_model_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

# ─────────────────────────────────────────────────────
# EVAL CHART 2 — Accuracy Leaderboard (horizontal bar)
# ─────────────────────────────────────────────────────
acc_series = metrics_df["Accuracy"].sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Eval Chart 2 — Accuracy Leaderboard", fontsize=16,
             fontweight="bold", color="#e0e0e0")

colors_sorted = [MODEL_COLORS[m] for m in acc_series.index]
bars = ax.barh(acc_series.index, acc_series.values,
               color=colors_sorted, edgecolor="#0f1117", height=0.55)
ax.set_xlim(0, 1.1)
ax.set_xlabel("Accuracy")

for bar, val in zip(bars, acc_series.values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%", va="center", ha="left",
            fontsize=12, fontweight="bold", color="#e0e0e0")

ax.axvline(acc_series.max(), color="#FF9F43", linestyle="--",
           linewidth=1.5, alpha=0.7, label=f"Best: {acc_series.max()*100:.1f}%")
ax.legend()
plt.tight_layout()
plt.savefig("eval2_accuracy_leaderboard.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

# ─────────────────────────────────────────────────────
# EVAL CHART 3 — Confusion Matrix (best model by F1)
# ─────────────────────────────────────────────────────
best_name = metrics_df["F1"].idxmax()
best_clf  = trained[best_name]
y_pred_best = best_clf.predict(X_test_sc)

cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(f"Eval Chart 3 — Confusion Matrix\n{best_name} (Best F1)",
             fontsize=14, fontweight="bold", color="#e0e0e0")

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Not Placed", "Placed"])
disp.plot(ax=ax, cmap="PuBu", colorbar=True)
ax.set_title("")
plt.tight_layout()
plt.savefig("eval3_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()
print(f"\n--- Classification Report: {best_name} ---")
print(classification_report(y_test, y_pred_best, target_names=["Not Placed", "Placed"]))

# ─────────────────────────────────────────────────────
# EVAL CHART 4 — ROC Curves (all models overlaid)
# ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Eval Chart 4 — ROC Curves (All Models)", fontsize=16,
             fontweight="bold", color="#e0e0e0")

for name, clf in trained.items():
    y_prob = clf.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2.5,
            label=f"{name}  (AUC = {roc_auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Random (AUC = 0.500)")
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("eval4_roc_curves.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

# ─────────────────────────────────────────────────────
# EVAL CHART 5 — Random Forest Feature Importances
# ─────────────────────────────────────────────────────
rf_clf      = trained["Random Forest"]
importances = rf_clf.feature_importances_
feat_df     = pd.DataFrame({
    "Feature": FEATURES, "Importance": importances
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Eval Chart 5 — Feature Importances (Random Forest)",
             fontsize=16, fontweight="bold", color="#e0e0e0")

norm = plt.Normalize(feat_df["Importance"].min(), feat_df["Importance"].max())
colors_imp = [plt.cm.plasma(norm(v)) for v in feat_df["Importance"]]
ax.barh(feat_df["Feature"], feat_df["Importance"],
        color=colors_imp, edgecolor="#0f1117", height=0.65)
for val, label in zip(feat_df["Importance"], feat_df["Feature"]):
    ax.text(val + 0.002, label, f"{val:.4f}", va="center",
            fontsize=10, color="#e0e0e0")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("eval5_feature_importance.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

print("\nAll 4 evaluation charts generated and saved.")
print(f"Best model (F1): {best_name}")
print("Run app.py for interactive dashboard.")
