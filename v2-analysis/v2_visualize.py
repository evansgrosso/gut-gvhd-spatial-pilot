import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "outputs", "scores", "block_scores_80um.parquet")
OUT_DIR = os.path.join(HERE, "outputs", "analysis")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Reads the same inputs v2_grade_comparison.py used, so run that first.
d = pd.read_parquet(SCORES)
epi = d[d["prop_epithelium"] >= d["prop_epithelium"].median()]
gradient = pd.read_csv(os.path.join(OUT_DIR, "step3_stem_gradient.csv"))

# Samples from the same patient are averaged so no patient is counted twice.
by_patient = gradient.groupby(["panel", "patient"])[
    ["diff", "pearson_r"]].mean().reset_index()
PANELS = list(gradient["panel"].unique())

# Box plus jittered points, so group overlap stays visible.
ncol = int(np.ceil(len(PANELS) / 2))
fig, axes = plt.subplots(2, ncol, figsize=(3 * ncol, 7))
for ax, panel in zip(axes.ravel(), PANELS):
    sns.boxplot(data=epi, x="grade_group", y=panel, ax=ax,
                order=["mild_no", "severe"], showfliers=False,
                palette=["#005A8F", "#B85000"], hue="grade_group",
                legend=False)
    sns.stripplot(data=epi.sample(min(len(epi), 2000), random_state=0),
                  x="grade_group", y=panel, ax=ax,
                  order=["mild_no", "severe"], color="black",
                  alpha=0.15, size=1.5, jitter=True)
    ax.set_title(panel, fontsize=9)
    ax.set_xlabel("")
for ax in axes.ravel()[len(PANELS):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "grade_comparison.png"), dpi=200)

# One point per patient, matching the 5 independent units the sign test uses.
fig, ax = plt.subplots(figsize=(9, 5))
sns.stripplot(data=by_patient, x="panel", y="diff", ax=ax, size=7,
              color="#005A8F")
ax.axhline(0, color="grey", lw=1, ls="--")
ax.set_ylabel("mean(stem-high third) - mean(stem-low third),\n"
              "one point per patient")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "stem_gradient_by_patient.png"), dpi=200)

print(f"Figures -> {FIG_DIR}")
