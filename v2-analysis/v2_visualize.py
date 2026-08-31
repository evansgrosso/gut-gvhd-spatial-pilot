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

# The seven measurements the focused comparison reports.
PANELS = ["OXPHOS", "GLUCOMETABOLISM", "GLUCO - OXPHOS",
          "MHC_I", "MHC_II_core", "CD74", "IFN_gamma"]

# Same ISC-rich stratum v2_grade_comparison.py tests on.
d = pd.read_parquet(SCORES)
cut = d.groupby("sample", observed=True)["stem_fraction"].transform(
    lambda s: s.quantile(2 / 3))
isc = d[d["stem_fraction"] >= cut]

# Box plus jittered points, so group overlap stays visible.
fig, axes = plt.subplots(2, 4, figsize=(12, 7))
for ax, panel in zip(axes.ravel(), PANELS):
    sns.boxplot(data=isc, x="grade_group", y=panel, ax=ax,
                order=["mild_no", "severe"], showfliers=False,
                palette=["#005A8F", "#B85000"], hue="grade_group",
                legend=False)
    sns.stripplot(data=isc.sample(min(len(isc), 2000), random_state=0),
                  x="grade_group", y=panel, ax=ax,
                  order=["mild_no", "severe"], color="black",
                  alpha=0.15, size=1.5, jitter=True)
    ax.set_title(panel, fontsize=9)
    ax.set_xlabel("")
for ax in axes.ravel()[len(PANELS):]:
    ax.axis("off")
fig.suptitle(f"Mild vs severe in ISC-rich bins "
             f"({len(isc)} of {len(d)} blocks, top stem_fraction third per sample)",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "grade_comparison.png"), dpi=200)

print(f"Figures -> {FIG_DIR}")
