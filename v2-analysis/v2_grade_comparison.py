"""
Hypothesis tests on the 80um block scores, from first principles.

Two tools, both standard first-course statistics:

  - Student's t-test (independent samples), for comparing two separate
    groups of patients (Step 2: severe vs mild/ND).
  - The sign test (a binomial test on how many patients agree in
    direction), for a result that is measured once per sample -- Step 3
    (does this score rise with stem_fraction, in this sample?) and Step 4
    (does this gene pair correlate positively, in this sample?) -- combined
    across patients.

No rank-based effect sizes, no partial correlation, no permutation nulls,
no multiple-testing correction across the panel list. An earlier version of
this analysis used those; they were built to work around specific problems
(a 7.6x depth range across samples, and 11,555 blocks that are not
independent observations). The sign test sidesteps the second problem
directly, since it is a test over patients, never over blocks -- so most of
that machinery was compensating for pseudoreplication that this design
avoids by construction.

Design constraint that no choice of test removes: there are 8 samples but
only 5 patients (C159, C162, and C179 each contribute two samples), and
grade maps onto exactly 2 patients (C159, C98 -- severe) versus 3 (C162,
C179, ND001 -- mild/ND). The smallest possible two-sided sign-test p-value
at n=5 is 2/2^5 = 0.0625, and a t-test at n=2 vs 3 has very little power
regardless of the true effect. So the number to read is the effect size (a
mean difference, in standard-deviation units since every score is already
z-scored) and how many of the patients agree in direction -- the p-value is
a floor check, not a measure of how strong the evidence is.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "outputs", "scores", "block_scores_80um.parquet")
OUT_DIR = os.path.join(HERE, "outputs", "analysis")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PANELS = ["GLUCOMETABOLISM", "Glycolysis", "PPP", "Pyr_to_TCA", "OXPHOS",
          "FAO", "TCA", "Fuel_contrast", "MHC_I", "MHCI_const", "MHCI_proc",
          "MHC_II_core", "CD74", "IFN_gamma"]


def load():
    if not os.path.exists(SCORES):
        raise SystemExit(
            f"Not found: {SCORES}\n"
            "Run v2_gene_scores.py on Stokes first, then pull the parquet:\n"
            "  scp stokes.ist.ucf.edu:/home/ev039784/gut-gvhd-spatial-pilot/"
            "v2-analysis/outputs/scores/block_scores_80um.parquet \\\n"
            f"     {os.path.dirname(SCORES)}/\n")
    d = pd.read_parquet(SCORES)
    if "grade_group" not in d.columns or "stem_fraction" not in d.columns:
        raise SystemExit(
            "This parquet is missing grade_group or stem_fraction -- it may "
            "predate the metadata fix. Re-run v2_gene_scores.py and pull it "
            "again.")
    return d


def patient_means(df, col):
    """Collapse blocks -> one mean per patient. This is the unit the
    between-group test (Step 2) actually runs on: 5 numbers, not 11,555."""
    return df.groupby("patient", observed=True)[col].mean()


def two_group_test(a, b):
    """Independent-samples t-test. a, b: arrays of per-unit means (block,
    sample, or patient). Returns (mean difference, t-test p-value)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    diff = a.mean() - b.mean()
    p = ttest_ind(a, b, equal_var=False).pvalue
    return diff, p


def sign_test(values):
    """Binomial test on how many values are positive vs negative. values:
    one number per independent unit (one per patient). Returns
    (n_positive, n_total, two-sided p)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    v = v[v != 0]
    if len(v) == 0:
        return 0, 0, np.nan
    k = int((v > 0).sum())
    return k, len(v), binomtest(k, len(v), 0.5).pvalue


d = load()
print(f"{len(d)} blocks, {d['sample'].nunique()} samples, "
      f"{d['patient'].nunique()} patients")

# Gate: the deconvolution has to put gastric cell types in gastric samples,
# or nothing built on stem_fraction below means anything.
print("\n=== Deconvolution check ===")
cols = [c for c in ("prop_Intestine Epithelial Stem cells",
                    "prop_Stomach Stem cells",
                    "prop_Instestinal Epithelial cells",
                    "prop_Stomach Epithelial cells") if c in d.columns]
chk = d.groupby(["sample", "tissue_class"], observed=True)[cols].mean()
print(chk.round(3).to_string())
print("\nExpect gastric samples to load on Stomach columns and intestinal "
      "samples on Intestinal ones. If not, stop here.")

# Primary stratum: epithelium-dominant blocks. Both the metabolic and MHC
# questions are about epithelium specifically.
if "prop_epithelium" in d.columns:
    cut = d["prop_epithelium"].median()
    epi = d[d["prop_epithelium"] >= cut].copy()
else:
    epi = d.copy()
print(f"\nEpithelium-dominant stratum: {len(epi)} of {len(d)} blocks")

# ---------------------------------------------------------------- Step 2
# Severe vs mild/ND is a comparison between two separate groups of
# patients (not a paired, within-patient measurement), so the fundamental
# tool is a plain two-sample t-test. Reported at three levels so the effect
# of pseudoreplication is visible directly: block-level pools 5,778 rows
# from just 5 patients and its p-value should not be trusted as evidence,
# even though it is printed for reference.
print("\n=== Step 2: Severe vs Mild/ND (epithelium-dominant blocks) ===")
print("diff = mean(severe) - mean(mild_no), in SD units (scores are "
      "z-scored per sample).")
print("Patient-level p cannot go below 2/2^5 = 0.0625 -- read diff, not p.\n")

rows = []
for panel in PANELS:
    if panel not in epi.columns:
        continue
    sev_blk = epi[epi.grade_group == "severe"][panel]
    mld_blk = epi[epi.grade_group == "mild_no"][panel]
    d_blk, p_blk = two_group_test(sev_blk, mld_blk)

    by_sample = epi.groupby("sample", observed=True)[panel].mean()
    g_sample = epi.groupby("sample", observed=True)["grade_group"].first()
    d_smp, p_smp = two_group_test(by_sample[g_sample == "severe"],
                                  by_sample[g_sample == "mild_no"])

    by_patient = patient_means(epi, panel)
    g_patient = epi.groupby("patient", observed=True)["grade_group"].first()
    d_pat, p_pat = two_group_test(by_patient[g_patient == "severe"],
                                  by_patient[g_patient == "mild_no"])

    rows.append(dict(panel=panel, diff_block=d_blk, p_block=p_blk,
                     diff_sample=d_smp, p_sample=p_smp,
                     diff_patient=d_pat, p_patient=p_pat))

grade = pd.DataFrame(rows)
print(grade.round(4).to_string(index=False))
print("\nblock-level and sample-level p-values are shown for reference only "
      "-- both re-count the same 5 patients many times over (2,289 blocks "
      "per patient on average), which makes their p-values far smaller than "
      "the data actually supports. Only the patient-level column is a valid "
      "test.")

# Tissue-stratified check, since grade is entangled with tissue: severe is
# 2 gastric + 1 intestinal patients, mild/no is 2 gastric + 3 intestinal.
t_rows = []
for tc, sub in epi.groupby("tissue_class", observed=True):
    for panel in PANELS:
        if panel not in sub.columns or sub.grade_group.nunique() < 2:
            continue
        diff, p = two_group_test(sub[sub.grade_group == "severe"][panel],
                                 sub[sub.grade_group == "mild_no"][panel])
        t_rows.append(dict(tissue_class=tc, panel=panel, diff=diff, p=p))
tissue = pd.DataFrame(t_rows)
if not tissue.empty:
    print("\n=== Step 2b: same comparison within tissue class (block-level) ===")
    print(tissue.pivot(index="panel", columns="tissue_class",
                       values="diff").round(3).to_string())

# ---------------------------------------------------------------- Step 3
# For each sample, split its blocks into the top third and bottom third by
# stem_fraction and take the difference in mean score -- the simplest
# possible way to ask "is this score higher in stem-rich or stem-poor
# tissue, in this sample?" A plain Pearson correlation is also reported for
# readers used to that number; the two nearly always agree; the tertile
# difference is treated as primary because it doesn't assume a straight-
# line relationship.
#
# One sample per patient would be ideal; three patients (C159, C162, C179)
# contribute two samples each. Those are averaged to one number per patient
# before the sign test, so no patient is counted twice.
print("\n=== Step 3: score vs stem_fraction, within each sample ===")
print("diff = mean(top third by stem_fraction) - mean(bottom third)\n")

grad_rows = []
for panel in PANELS:
    if panel not in d.columns:
        continue
    for sid, sub in d.groupby("sample", observed=True):
        sub = sub.dropna(subset=["stem_fraction", panel])
        if len(sub) < 30:
            continue
        lo, hi = sub["stem_fraction"].quantile([1 / 3, 2 / 3])
        diff = (sub.loc[sub.stem_fraction >= hi, panel].mean()
                - sub.loc[sub.stem_fraction <= lo, panel].mean())
        r, _ = pearsonr(sub[panel], sub["stem_fraction"])
        grad_rows.append(dict(panel=panel, sample=sid,
                              patient=sub["patient"].iloc[0],
                              tissue_class=sub["tissue_class"].iloc[0],
                              diff=diff, pearson_r=r))

gradient = pd.DataFrame(grad_rows)
by_patient = gradient.groupby(["panel", "patient"], observed=True)[
    ["diff", "pearson_r"]].mean().reset_index()

summary = []
for panel, sub in by_patient.groupby("panel", observed=True):
    k, n, p = sign_test(sub["diff"])
    summary.append(dict(panel=panel, median_diff=sub["diff"].median(),
                        median_pearson_r=sub["pearson_r"].median(),
                        n_patients_positive=k, n_patients=n, sign_test_p=p))
summary = pd.DataFrame(summary).sort_values("panel")
print(summary.round(4).to_string(index=False))
print(f"\nn_patients_positive / n_patients is the number of independent "
      f"patients that agree in\ndirection -- {5}/5 or {4}/5 is the strongest "
      f"claim this design can make. sign_test_p is the\ntwo-sided binomial "
      f"p-value for that count; it cannot go below 0.0625 at n=5.")

intest = by_patient.merge(
    gradient[["patient", "tissue_class"]].drop_duplicates(), on="patient")
intest = intest[intest.tissue_class == "intestinal"]
if not intest.empty:
    print("\nIntestinal-only patients (C162, C159, C179, ND001 where "
          "intestinal), median diff:")
    print(intest.groupby("panel", observed=True)["diff"].median()
          .round(3).to_string())

# ---------------------------------------------------------------- Step 4
# Same logic as Step 3, applied to a gene-gene correlation instead of a
# score-vs-stem_fraction one. This step needs no cell-type proportions at
# all -- it is pure gene expression -- so it is the most assumption-free
# check in this analysis.
PAIRS = [("CD74", "MHC_II_core"), ("MHC_II_core", "IFN_gamma"),
         ("CD74", "IFN_gamma"), ("MHCI_proc", "IFN_gamma"),
         ("MHCI_const", "IFN_gamma"), ("MHCI_proc", "MHCI_const")]

rep_rows = []
for a, b in PAIRS:
    if a not in d.columns or b not in d.columns:
        continue
    for sid, sub in d.groupby("sample", observed=True):
        sub = sub.dropna(subset=[a, b])
        r, _ = pearsonr(sub[a], sub[b])
        rep_rows.append(dict(pair=f"{a} ~ {b}", sample=sid,
                             patient=sub["patient"].iloc[0], r=r))
replication = pd.DataFrame(rep_rows)
rep_by_patient = replication.groupby(["pair", "patient"], observed=True)[
    "r"].mean().reset_index()

print("\n=== Step 4: per-sample gene-gene replication ===")
rep_summary = []
for pair, sub in rep_by_patient.groupby("pair", observed=True):
    k, n, p = sign_test(sub["r"])
    rep_summary.append(dict(pair=pair, median_r=sub["r"].median(),
                            n_patients_positive=k, n_patients=n,
                            sign_test_p=p))
rep_summary = pd.DataFrame(rep_summary)
print(rep_summary.round(4).to_string(index=False))
print("\nCD74 ~ MHC_II_core is the key row -- CIITA transcriptionally drives "
      "CD74, so a persistently\nweak correlation means CD74 is not reporting "
      "MHC-II biology in this assay.")

# ---------------------------------------------------------------- outputs
grade.to_csv(os.path.join(OUT_DIR, "step2_grade_comparison.csv"), index=False)
if not tissue.empty:
    tissue.to_csv(os.path.join(OUT_DIR, "step2b_by_tissue.csv"), index=False)
gradient.to_csv(os.path.join(OUT_DIR, "step3_stem_gradient.csv"), index=False)
summary.to_csv(os.path.join(OUT_DIR, "step3_summary_by_patient.csv"), index=False)
replication.to_csv(os.path.join(OUT_DIR, "step4_replication.csv"), index=False)
rep_summary.to_csv(os.path.join(OUT_DIR, "step4_summary_by_patient.csv"),
                   index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Box plus jittered points, so the overlap between groups (which is
    # substantial almost everywhere) stays visible rather than being
    # hidden behind a five-number summary.
    keep = [p for p in PANELS if p in epi.columns]
    fig, axes = plt.subplots(2, int(np.ceil(len(keep) / 2)),
                             figsize=(3 * np.ceil(len(keep) / 2), 7))
    for ax, panel in zip(axes.ravel(), keep):
        sns.boxplot(data=epi, x="grade_group", y=panel, ax=ax,
                    order=["mild_no", "severe"], showfliers=False,
                    palette=["#005A8F", "#B85000"], hue="grade_group",
                    legend=False)
        sub = epi.sample(min(len(epi), 2000), random_state=0)
        sns.stripplot(data=sub, x="grade_group", y=panel, ax=ax,
                      order=["mild_no", "severe"], color="black",
                      alpha=0.15, size=1.5, jitter=True)
        ax.set_title(panel, fontsize=9)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(keep):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "grade_comparison.png"), dpi=200)

    # One point per patient (samples from the same patient averaged), so
    # the plot shows exactly the 5 independent units the sign test uses.
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.stripplot(data=by_patient, x="panel", y="diff", ax=ax, size=7,
                  color="#005A8F")
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.set_ylabel("mean(stem-high third) - mean(stem-low third),\n"
                  "one point per patient")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "stem_gradient_by_patient.png"), dpi=200)
    print(f"\nFigures -> {FIG_DIR}")
except Exception as exc:
    print(f"\nPlotting skipped: {exc}")

print(f"Tables  -> {OUT_DIR}")
