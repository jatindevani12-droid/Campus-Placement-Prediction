"""
CAMPUS PLACEMENT — Step 4: Rich Visualizations (10 charts)
Run this file standalone to see all charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ── Style setup ────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="husl")
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#444",
})

ACCENT   = ["#6C63FF", "#FF6584"]
PALETTE  = ["#6C63FF", "#FF6584", "#43C59E", "#FF9F43", "#54A0FF"]
PLACED_C = "#6C63FF"
NOP_C    = "#FF6584"

# ── Load & preprocess ──────────────────────────────────────────
df_raw = pd.read_csv("placement.csv")
df = df_raw.copy()
df.drop(columns=["sl_no"], inplace=True)
df["salary"].fillna(0, inplace=True)

le = LabelEncoder()
for col in ["ssc_b", "hsc_b", "hsc_s", "degree_t", "specialisation"]:
    df[col] = le.fit_transform(df[col])
df["gender_code"] = df["gender"].map({"M": 0, "F": 1})
df["workex_code"] = df["workex"].map({"No": 0, "Yes": 1})
df["status_code"] = df["status"].map({"Not Placed": 0, "Placed": 1})


def savefig_and_show(title):
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').replace('/', '-')}.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.show()


# ══════════════════════════════════════════════════════════════
# CHART 1 — Placement Distribution (Donut + Count)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Chart 1 — Placement Distribution", fontsize=16, fontweight="bold", color="#e0e0e0", y=1.02)

counts = df["status"].value_counts()
ax = axes[0]
wedges, texts, autotexts = ax.pie(
    counts, labels=counts.index, autopct="%1.1f%%",
    colors=[PLACED_C, NOP_C], startangle=90,
    wedgeprops=dict(width=0.55, edgecolor="#0f1117", linewidth=2),
    textprops={"color": "#e0e0e0", "fontsize": 12}
)
for at in autotexts:
    at.set_fontsize(13); at.set_fontweight("bold")
ax.set_title("Placement Share", color="#e0e0e0", pad=12)

ax2 = axes[1]
bars = ax2.bar(counts.index, counts.values, color=[PLACED_C, NOP_C],
               width=0.5, edgecolor="#0f1117", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             str(val), ha="center", va="bottom", fontsize=13, fontweight="bold", color="#e0e0e0")
ax2.set_title("Student Counts", color="#e0e0e0")
ax2.set_ylabel("Number of Students")
savefig_and_show("chart1_placement_distribution")


# ══════════════════════════════════════════════════════════════
# CHART 2 — Placement by Gender
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Chart 2 — Placement by Gender", fontsize=16, fontweight="bold", color="#e0e0e0")

gender_status = df.groupby(["gender", "status"]).size().unstack(fill_value=0)
gender_status.plot(kind="bar", ax=ax, color=[NOP_C, PLACED_C],
                   edgecolor="#0f1117", width=0.55, rot=0)
ax.set_xlabel("Gender"); ax.set_ylabel("Count")
ax.legend(title="Status", labels=["Not Placed", "Placed"])
for container in ax.containers:
    ax.bar_label(container, padding=4, color="#e0e0e0", fontsize=11)
savefig_and_show("chart2_placement_by_gender")


# ══════════════════════════════════════════════════════════════
# CHART 3 — Work Experience vs Placement
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Chart 3 — Work Experience vs Placement", fontsize=16, fontweight="bold", color="#e0e0e0")

we_status = df.groupby(["workex", "status"]).size().unstack(fill_value=0)
we_status.plot(kind="bar", ax=ax, color=[NOP_C, PLACED_C],
               edgecolor="#0f1117", width=0.55, rot=0)
ax.set_xlabel("Work Experience"); ax.set_ylabel("Count")
ax.legend(title="Status", labels=["Not Placed", "Placed"])
for container in ax.containers:
    ax.bar_label(container, padding=4, color="#e0e0e0", fontsize=11)
savefig_and_show("chart3_workex_vs_placement")


# ══════════════════════════════════════════════════════════════
# CHART 4 — HSC Stream vs Placement
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Chart 4 — HSC Stream vs Placement", fontsize=16, fontweight="bold", color="#e0e0e0")

hsc_status = df.groupby(["hsc_s", "status"]).size().unstack(fill_value=0)
hsc_status.plot(kind="bar", ax=ax, color=[NOP_C, PLACED_C],
                edgecolor="#0f1117", width=0.55, rot=0)
ax.set_xlabel("HSC Stream"); ax.set_ylabel("Count")
ax.legend(title="Status", labels=["Not Placed", "Placed"])
for container in ax.containers:
    ax.bar_label(container, padding=4, color="#e0e0e0", fontsize=11)
savefig_and_show("chart4_hsc_stream_vs_placement")


# ══════════════════════════════════════════════════════════════
# CHART 5 — MBA Specialisation vs Placement
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Chart 5 — MBA Specialisation vs Placement", fontsize=16, fontweight="bold", color="#e0e0e0")

spec_status = df.groupby(["specialisation", "status"]).size().unstack(fill_value=0)
spec_status.plot(kind="bar", ax=ax, color=[NOP_C, PLACED_C],
                 edgecolor="#0f1117", width=0.55, rot=0)
ax.set_xlabel("Specialisation"); ax.set_ylabel("Count")
ax.legend(title="Status", labels=["Not Placed", "Placed"])
for container in ax.containers:
    ax.bar_label(container, padding=4, color="#e0e0e0", fontsize=11)
savefig_and_show("chart5_specialisation_vs_placement")


# ══════════════════════════════════════════════════════════════
# CHART 6 — Score Distributions (KDE)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Chart 6 — Score Distributions by Placement Status", fontsize=16, fontweight="bold", color="#e0e0e0")

score_cols = [("ssc_p", "SSC %"), ("hsc_p", "HSC %"), ("degree_p", "Degree %")]
for ax, (col, label) in zip(axes, score_cols):
    for status, color, name in [("Placed", PLACED_C, "Placed"), ("Not Placed", NOP_C, "Not Placed")]:
        subset = df_raw[df_raw["status"] == status][col]
        ax.hist(subset, bins=15, alpha=0.4, color=color, edgecolor=color, density=True)
        subset.plot.kde(ax=ax, color=color, linewidth=2.5, label=name)
    ax.set_title(label); ax.set_xlabel(label); ax.set_ylabel("Density")
    ax.legend()
savefig_and_show("chart6_score_distributions")


# ══════════════════════════════════════════════════════════════
# CHART 7 — Box Plots: Scores vs Status
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Chart 7 — Box Plots: Academic Scores vs Placement", fontsize=16, fontweight="bold", color="#e0e0e0")

for ax, (col, label) in zip(axes, score_cols):
    sns.boxplot(
        data=df_raw, x="status", y=col, ax=ax,
        palette={"Placed": PLACED_C, "Not Placed": NOP_C},
        width=0.5, linewidth=1.5,
        order=["Not Placed", "Placed"],
        boxprops=dict(alpha=0.8)
    )
    ax.set_title(label); ax.set_xlabel(""); ax.set_ylabel(label)
savefig_and_show("chart7_boxplots")


# ══════════════════════════════════════════════════════════════
# CHART 8 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 8))
fig.suptitle("Chart 8 — Feature Correlation Heatmap", fontsize=16, fontweight="bold", color="#e0e0e0")

num_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p", "salary", "status_code"]
corr = df[num_cols].rename(columns={"status_code": "status"}).corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(260, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f",
            linewidths=0.5, linecolor="#0f1117", ax=ax,
            annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
ax.set_title("Pearson Correlation", pad=12, color="#e0e0e0")
savefig_and_show("chart8_heatmap")


# ══════════════════════════════════════════════════════════════
# CHART 9 — Salary Distribution (Placed Students Only)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chart 9 — Salary Distribution (Placed Students)", fontsize=16, fontweight="bold", color="#e0e0e0")

salary_df = df_raw[df_raw["status"] == "Placed"]["salary"]

# Histogram + KDE
ax = axes[0]
ax.hist(salary_df / 1000, bins=20, color=PLACED_C, edgecolor="#6C63FF", alpha=0.6, density=True)
(salary_df / 1000).plot.kde(ax=ax, color="#FF9F43", linewidth=2.5)
ax.set_xlabel("Salary (₹ thousands)"); ax.set_ylabel("Density")
ax.set_title("Distribution")
ax.axvline(salary_df.mean() / 1000, color="#FF6584", linestyle="--", linewidth=2, label=f"Mean ₹{salary_df.mean()/1000:.0f}k")
ax.legend()

# Box plot
ax2 = axes[1]
ax2.boxplot(salary_df / 1000, patch_artist=True,
            boxprops=dict(facecolor=PLACED_C, color="#e0e0e0", alpha=0.7),
            medianprops=dict(color="#FF9F43", linewidth=2.5),
            whiskerprops=dict(color="#aaa"),
            capprops=dict(color="#aaa"),
            flierprops=dict(marker="o", color="#FF6584", markersize=6))
ax2.set_ylabel("Salary (₹ thousands)"); ax2.set_title("Box Plot")
savefig_and_show("chart9_salary_distribution")


# ══════════════════════════════════════════════════════════════
# CHART 10 — Pair Plot (key features, colored by status)
# ══════════════════════════════════════════════════════════════
print("Chart 10 — Generating pair plot (may take a moment)…")
pair_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p", "status"]
pair_df   = df_raw[pair_cols].copy()

g = sns.pairplot(
    pair_df, hue="status", corner=True,
    palette={"Placed": PLACED_C, "Not Placed": NOP_C},
    plot_kws={"alpha": 0.55, "s": 30},
    diag_kws={"alpha": 0.5}
)
g.figure.suptitle("Chart 10 — Pair Plot: Key Academic Features", y=1.02,
                  fontsize=16, fontweight="bold")
plt.savefig("chart10_pairplot.png", dpi=120, bbox_inches="tight", facecolor="white")
plt.show()

print("\n✅ All 10 charts generated and saved as PNG files.")