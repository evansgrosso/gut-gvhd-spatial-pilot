import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "outputs", "scores", "block_scores_80um.parquet")
OUT_DIR = os.path.join(HERE, "outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# Score columns to test. Filtered to the ones the parquet actually has.
PANELS = ["GLUCOMETABOLISM", "Glycolysis", "PPP", "Pyr_to_TCA",
          "OXPHOS", "ETC_CI", "ETC_CIII", "ETC_CIV", "ETC_CV", "TCA",
          "FAO", "GLUCO - OXPHOS", "MHC_I", "MHCI_const", "MHCI_proc",
          "MHC_II_core", "CD74", "IFN_gamma"]

# Gene pairs checked for per-sample replication in step 4.
PAIRS = [("CD74", "MHC_II_core"), ("MHC_II_core", "IFN_gamma"),
         ("CD74", "IFN_gamma"), ("MHCI_proc", "IFN_gamma"),
         ("MHCI_const", "IFN_gamma"), ("MHCI_proc", "MHCI_const")]

# Independent-samples t-test. Returns the mean difference and its p-value.
def two_group_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    return a.mean() - b.mean(), ttest_ind(a, b, equal_var=False).pvalue

# Binomial test on how many patients agree in direction.
def sign_test(values):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    v = v[v != 0]
    k = int((v > 0).sum())
    return k, len(v), binomtest(k, len(v), 0.5).pvalue

d = pd.read_parquet(SCORES)
PANELS = [p for p in PANELS if p in d.columns]
PAIRS = [(a, b) for a, b in PAIRS if a in d.columns and b in d.columns]

# Gate: gastric cell types have to land in gastric samples.
GATE = ["prop_Intestine Epithelial Stem cells", "prop_Stomach Stem cells",
        "prop_Instestinal Epithelial cells", "prop_Stomach Epithelial cells"]
print("=== Deconvolution check ===")
print(d.groupby(["sample", "tissue_class"], observed=True)[GATE]
      .mean().round(3).to_string())

# Primary stratum: epithelium-dominant blocks.
epi = d[d["prop_epithelium"] >= d["prop_epithelium"].median()].copy()

# Step 2: severe vs mild/ND, at block, sample and patient level.
rows = []
for panel in PANELS:
    d_blk, p_blk = two_group_test(epi[epi.grade_group == "severe"][panel],
                                  epi[epi.grade_group == "mild_no"][panel])

    by_sample = epi.groupby("sample", observed=True)[panel].mean()
    g_sample = epi.groupby("sample", observed=True)["grade_group"].first()
    d_smp, p_smp = two_group_test(by_sample[g_sample == "severe"],
                                  by_sample[g_sample == "mild_no"])

    by_pat = epi.groupby("patient", observed=True)[panel].mean()
    g_pat = epi.groupby("patient", observed=True)["grade_group"].first()
    d_pat, p_pat = two_group_test(by_pat[g_pat == "severe"],
                                  by_pat[g_pat == "mild_no"])

    rows.append(dict(panel=panel, diff_block=d_blk, p_block=p_blk,
                     diff_sample=d_smp, p_sample=p_smp,
                     diff_patient=d_pat, p_patient=p_pat))
grade = pd.DataFrame(rows)
print("\n=== Step 2: Severe vs Mild/ND ===")
print(grade.round(4).to_string(index=False))

# Step 2b: same comparison within tissue class, since grade is entangled with tissue.
t_rows = []
for tc, sub in epi.groupby("tissue_class", observed=True):
    for panel in PANELS:
        diff, p = two_group_test(sub[sub.grade_group == "severe"][panel],
                                 sub[sub.grade_group == "mild_no"][panel])
        t_rows.append(dict(tissue_class=tc, panel=panel, diff=diff, p=p))
tissue = pd.DataFrame(t_rows)
print("\n=== Step 2b: within tissue class ===")
print(tissue.pivot(index="panel", columns="tissue_class",
                   values="diff").round(3).to_string())

# Step 3: top third minus bottom third by stem_fraction, within each sample.
grad_rows = []
for panel in PANELS:
    for sid, sub in d.groupby("sample", observed=True):
        sub = sub.dropna(subset=["stem_fraction", panel])
        lo, hi = sub["stem_fraction"].quantile([1 / 3, 2 / 3])
        diff = (sub.loc[sub.stem_fraction >= hi, panel].mean()
                - sub.loc[sub.stem_fraction <= lo, panel].mean())
        r, _ = pearsonr(sub[panel], sub["stem_fraction"])
        grad_rows.append(dict(panel=panel, sample=sid,
                              patient=sub["patient"].iloc[0],
                              tissue_class=sub["tissue_class"].iloc[0],
                              diff=diff, pearson_r=r))
gradient = pd.DataFrame(grad_rows)

# Samples from the same patient are averaged so no patient is counted twice.
by_patient = gradient.groupby(["panel", "patient"], observed=True)[
    ["diff", "pearson_r"]].mean().reset_index()

summary = []
for panel, sub in by_patient.groupby("panel", observed=True):
    k, n, p = sign_test(sub["diff"])
    summary.append(dict(panel=panel, median_diff=sub["diff"].median(),
                        median_pearson_r=sub["pearson_r"].median(),
                        n_patients_positive=k, n_patients=n, sign_test_p=p))
summary = pd.DataFrame(summary).sort_values("panel")
print("\n=== Step 3: score vs stem_fraction ===")
print(summary.round(4).to_string(index=False))

intest = by_patient.merge(
    gradient[["patient", "tissue_class"]].drop_duplicates(), on="patient")
intest = intest[intest.tissue_class == "intestinal"]
print("\nIntestinal-only patients, median diff:")
print(intest.groupby("panel", observed=True)["diff"].median().round(3).to_string())

# Step 4: per-sample gene-gene correlation, using no cell-type proportions.
rep_rows = []
for a, b in PAIRS:
    for sid, sub in d.groupby("sample", observed=True):
        sub = sub.dropna(subset=[a, b])
        r, _ = pearsonr(sub[a], sub[b])
        rep_rows.append(dict(pair=f"{a} ~ {b}", sample=sid,
                             patient=sub["patient"].iloc[0], r=r))
replication = pd.DataFrame(rep_rows)
rep_by_patient = replication.groupby(["pair", "patient"], observed=True)[
    "r"].mean().reset_index()

rep_summary = []
for pair, sub in rep_by_patient.groupby("pair", observed=True):
    k, n, p = sign_test(sub["r"])
    rep_summary.append(dict(pair=pair, median_r=sub["r"].median(),
                            n_patients_positive=k, n_patients=n,
                            sign_test_p=p))
rep_summary = pd.DataFrame(rep_summary)
print("\n=== Step 4: gene-gene replication ===")
print(rep_summary.round(4).to_string(index=False))

grade.to_csv(os.path.join(OUT_DIR, "step2_grade_comparison.csv"), index=False)
tissue.to_csv(os.path.join(OUT_DIR, "step2b_by_tissue.csv"), index=False)
gradient.to_csv(os.path.join(OUT_DIR, "step3_stem_gradient.csv"), index=False)
summary.to_csv(os.path.join(OUT_DIR, "step3_summary_by_patient.csv"), index=False)
replication.to_csv(os.path.join(OUT_DIR, "step4_replication.csv"), index=False)
rep_summary.to_csv(os.path.join(OUT_DIR, "step4_summary_by_patient.csv"),
                   index=False)
