"""
Hypothesis tests on the 80um block scores.

Runs locally on block_scores_80um.parquet pulled from Stokes.

Three analyses:
  Step 2  Severe vs Mild/ND, tested at block, sample, and patient level
  Step 3  Score vs stem-fraction within sample, with a confound ladder
  Step 4  Whether CD74 diverges from the MHC-II transactivator core in all 8

Design constraints that shape how the output should be read:
  - 8 samples, 5 patients. C159 and C98 are severe; C162, C179, ND001 are
    mild/ND. So the patient-level test is 2 vs 3, where the smallest possible
    two-sided Mann-Whitney p is 2/C(5,2) = 0.2. It cannot reach significance at
    any effect size, and is reported for its effect size only.
  - Grade is entangled with tissue: severe is 2 gastric + 1 intestinal, mild/no
    is 2 gastric + 3 intestinal. Not adjustable at this n, only reportable.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "outputs", "scores", "block_scores_80um.parquet")
OUT_DIR = os.path.join(HERE, "outputs", "analysis")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PANELS = ["GLUCOMETABOLISM", "Glycolysis", "PPP", "Pyr_to_TCA", "OXPHOS",
          "FAO", "TCA", "Fuel_contrast", "MHC_I", "MHCI_const", "MHCI_proc",
          "MHC_II_core", "CD74", "IFN_gamma"]

# Covariates added one at a time, in the order Stage 8 used.
LADDER = ["depth", "prop_APC_total", "prop_Tcell_total", "IFN_gamma"]


def rank_biserial(a, b):
    """Effect size for Mann-Whitney. Does not inflate with sample size, which
    is why it is the headline number rather than the p-value."""
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def mw(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 1 or len(b) < 1:
        return np.nan, np.nan
    return mannwhitneyu(a, b, alternative="two-sided").pvalue, rank_biserial(a, b)


def partial_spearman(x, y, covars):
    """Spearman partial correlation of x and y given covars, on ranks."""
    M = np.column_stack([rankdata(x), rankdata(y)]
                        + [rankdata(c) for c in covars])
    ok = ~np.isnan(M).any(1)
    M = M[ok]
    if len(M) < 10:
        return np.nan
    C = np.corrcoef(M, rowvar=False)
    try:
        P = np.linalg.pinv(C)
    except np.linalg.LinAlgError:
        return np.nan
    return -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])


def load():
    if not os.path.exists(SCORES):
        raise SystemExit(
            f"Not found: {SCORES}\n"
            "Pull it from Stokes first:\n"
            "  scp ev039784@stokes:/home/ev039784/gut-gvhd-spatial-pilot/"
            "v2-analysis/outputs/scores/block_scores_80um.parquet \\\n"
            f"     {os.path.dirname(SCORES)}/\n")
    d = pd.read_parquet(SCORES)
    if "grade_group" not in d.columns:
        raise SystemExit(
            "This parquet predates the metadata fix -- no 'grade_group' column.\n"
            "Re-run the corrected v2_gene_scores.py on Stokes and pull it again.")
    prop_cols = [c for c in d.columns if c.startswith("prop_")]
    if "stem_fraction" not in d.columns or not prop_cols:
        raise SystemExit(
            "No cell-type proportions in this parquet "
            f"({len(prop_cols)} prop_ columns, "
            f"stem_fraction={'present' if 'stem_fraction' in d.columns else 'absent'}).\n"
            "The Starfysh proportions were loaded but their columns were not "
            "named, so every cell-type lookup found nothing.\n"
            "Re-run the corrected v2_gene_scores.py -- it now names obsm "
            "columns from GVHD_spatial_signature.csv and stops if it cannot.\n")
    return d


d = load()
print(f"{len(d)} blocks, {d['sample'].nunique()} samples, "
      f"{d['patient'].nunique()} patients")

# Gate: the deconvolution has to put gastric cell types in gastric samples.
print("\n=== Deconvolution check ===")
stem_i, stem_s = "prop_Intestine Epithelial Stem cells", "prop_Stomach Stem cells"
epi_i, epi_s = "prop_Instestinal Epithelial cells", "prop_Stomach Epithelial cells"
chk = d.groupby(["sample", "tissue_class"], observed=True)[
    [c for c in (stem_i, stem_s, epi_i, epi_s) if c in d.columns]].mean()
print(chk.round(3).to_string())
print("\nExpect gastric samples to load on Stomach columns and intestinal "
      "samples on Intestinal ones. If not, stop here -- the proportions are\n"
      "unreliable and Steps 2-4 are not interpretable.")

# Primary stratum: epithelium-dominant blocks, per Stage 5/6's IEC-rich
# restriction. Metabolic and MHC questions are both about epithelium.
if "prop_epithelium" in d.columns:
    cut = d["prop_epithelium"].median()
    epi = d[d["prop_epithelium"] >= cut].copy()
else:
    epi = d.copy()
print(f"\nEpithelium-dominant stratum: {len(epi)} of {len(d)} blocks")

# ---------------------------------------------------------------- Step 2
rows = []
for panel in PANELS:
    for suffix in ("", "_dadj"):
        col = panel + suffix
        if col not in epi.columns:
            continue
        sev = epi[epi.grade_group == "severe"][col]
        mld = epi[epi.grade_group == "mild_no"][col]
        p_blk, r_blk = mw(sev, mld)

        by_s = epi.groupby("sample", observed=True)[col].mean()
        g_s = epi.groupby("sample", observed=True)["grade_group"].first()
        p_smp, r_smp = mw(by_s[g_s == "severe"], by_s[g_s == "mild_no"])

        by_p = epi.groupby("patient", observed=True)[col].mean()
        g_p = epi.groupby("patient", observed=True)["grade_group"].first()
        p_pat, r_pat = mw(by_p[g_p == "severe"], by_p[g_p == "mild_no"])

        rows.append(dict(panel=panel, scores=suffix or "raw",
                         mean_severe=sev.mean(), mean_mild_no=mld.mean(),
                         r_block=r_blk, p_block=p_blk,
                         r_sample=r_smp, p_sample=p_smp,
                         r_patient=r_pat, p_patient=p_pat))

grade = pd.DataFrame(rows)
for lvl in ("block", "sample", "patient"):
    for sc in grade.scores.unique():
        m = grade.scores == sc
        ok = m & grade[f"p_{lvl}"].notna()
        grade.loc[ok, f"q_{lvl}"] = multipletests(
            grade.loc[ok, f"p_{lvl}"], method="fdr_bh")[1]

print("\n=== Step 2: Severe vs Mild/ND (epithelium-dominant blocks) ===")
print("r = rank-biserial; positive means HIGHER in severe.")
print("Patient-level p cannot go below 0.20 at n=2 vs 3 -- read r, not p.\n")
show = grade[grade.scores == "raw"][
    ["panel", "r_block", "q_block", "r_sample", "p_sample", "r_patient"]]
print(show.round(4).to_string(index=False))

# Does the conclusion survive swapping raw for depth-adjusted scores?
piv = grade.pivot(index="panel", columns="scores", values="r_block")
if "_dadj" in piv.columns:
    piv["flips_sign"] = np.sign(piv["raw"]) != np.sign(piv["_dadj"])
    print("\nRaw vs depth-adjusted block effect (sign flip = depth-driven):")
    print(piv.round(3).to_string())

# Tissue-stratified check on the grade/tissue entanglement.
t_rows = []
for tc, sub in epi.groupby("tissue_class", observed=True):
    for panel in PANELS:
        if panel not in sub.columns or sub.grade_group.nunique() < 2:
            continue
        p, r = mw(sub[sub.grade_group == "severe"][panel],
                  sub[sub.grade_group == "mild_no"][panel])
        t_rows.append(dict(tissue_class=tc, panel=panel, r_block=r, p_block=p))
tissue = pd.DataFrame(t_rows)
if not tissue.empty:
    print("\n=== Step 2b: same test within tissue class ===")
    print(tissue.pivot(index="panel", columns="tissue_class",
                       values="r_block").round(3).to_string())

# ---------------------------------------------------------------- Step 3
grad_rows = []
for panel in PANELS:
    if panel not in d.columns:
        continue
    for sid, sub in d.groupby("sample", observed=True):
        if sub["stem_fraction"].notna().sum() < 50:
            continue
        rec = dict(panel=panel, sample=sid,
                   tissue_class=sub["tissue_class"].iloc[0],
                   grade_group=sub["grade_group"].iloc[0],
                   rho_raw=spearmanr(sub[panel], sub["stem_fraction"],
                                     nan_policy="omit")[0])
        covars = []
        for cov in LADDER:
            if cov not in sub.columns or cov == panel:
                continue
            covars.append(sub[cov].values)
            rec[f"rho_+{cov}"] = partial_spearman(
                sub[panel].values, sub["stem_fraction"].values, covars)
        grad_rows.append(rec)

gradient = pd.DataFrame(grad_rows)
print("\n=== Step 3: score vs stem_fraction, within sample ===")
print("Median across the 8 samples, with the confound ladder:\n")
cols = ["rho_raw"] + [f"rho_+{c}" for c in LADDER if f"rho_+{c}" in gradient]
summ = gradient.groupby("panel", observed=True)[cols].median()
summ["n_samples"] = gradient.groupby("panel", observed=True).size()
summ["n_same_sign"] = gradient.groupby("panel", observed=True)["rho_raw"].apply(
    lambda s: int(max((s > 0).sum(), (s < 0).sum())))
print(summ.round(3).to_string())
print("\nn_same_sign is how many of the 8 samples agree in direction; 8/8 or "
      "7/8 is\nthe consistency claim, since 8 samples cannot support much more.")

intest = gradient[gradient.tissue_class == "intestinal"]
if not intest.empty:
    print("\nIntestinal-only stratum (SLV12/14/16/17), median rho_raw:")
    print(intest.groupby("panel", observed=True)["rho_raw"].median()
          .round(3).to_string())

# ---------------------------------------------------------------- Step 4
PAIRS = [("CD74", "MHC_II_core"), ("MHC_II_core", "IFN_gamma"),
         ("CD74", "IFN_gamma"), ("MHCI_proc", "IFN_gamma"),
         ("MHCI_const", "IFN_gamma"), ("MHCI_proc", "MHCI_const")]
rep_rows = []
for a, b in PAIRS:
    for suffix in ("", "_dadj"):
        ca, cb = a + suffix, b + suffix
        if ca not in d.columns or cb not in d.columns:
            continue
        rec = {"pair": f"{a} ~ {b}", "scores": suffix or "raw"}
        vals = {}
        for sid, sub in d.groupby("sample", observed=True):
            vals[sid] = spearmanr(sub[ca], sub[cb], nan_policy="omit")[0]
        v = np.array(list(vals.values()))
        # A pair that changes sign between samples is not a replicated finding,
        # however large its pooled value looks.
        rec.update(vals, median=np.median(v), lo=v.min(), hi=v.max(),
                   all_same_sign=len(set(np.sign(v))) == 1)
        rep_rows.append(rec)
replication = pd.DataFrame(rep_rows)

print("\n=== Step 4: per-sample replication ===")
print(replication[["pair", "scores", "median", "lo", "hi", "all_same_sign"]]
      .round(3).to_string(index=False))
print("\nRead the _dadj rows: both members of these pairs correlate with depth,")
print("so the raw values are inflated by depth they share. A pair that changes")
print("sign across samples has not replicated, whatever its pooled value.")
print("\nCD74 ~ MHC_II_core is the key row -- CIITA transcriptionally drives")
print("CD74, so a persistently weak correlation means CD74 is not reporting")
print("MHC-II biology in this assay.")

# ---------------------------------------------------------------- outputs
grade.to_csv(os.path.join(OUT_DIR, "step2_grade_comparison.csv"), index=False)
if not tissue.empty:
    tissue.to_csv(os.path.join(OUT_DIR, "step2b_by_tissue.csv"), index=False)
gradient.to_csv(os.path.join(OUT_DIR, "step3_stem_gradient.csv"), index=False)
replication.to_csv(os.path.join(OUT_DIR, "step4_replication.csv"), index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Box plus jittered points, so the overlap between groups stays visible
    # rather than being hidden behind a five-number summary.
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

    # Per-sample gradient effects, so between-sample spread is visible.
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.stripplot(data=gradient, x="panel", y="rho_raw", hue="tissue_class",
                  ax=ax, dodge=True, size=6)
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.set_ylabel("Spearman rho vs stem_fraction (one point per sample)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "stem_gradient_by_sample.png"), dpi=200)
    print(f"\nFigures -> {FIG_DIR}")
except Exception as exc:
    print(f"\nPlotting skipped: {exc}")

print(f"Tables  -> {OUT_DIR}")
