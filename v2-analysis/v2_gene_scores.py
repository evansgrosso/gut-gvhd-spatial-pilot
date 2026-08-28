"""
Gene signature scoring on 80um blocks, joined to Starfysh proportions.

Runs on Stokes, where adata_out.h5ad lives. Writes one compact parquet of
per-block scores + proportions that is small enough to move locally.

Panels were validated on all 8 samples by testing each one's internal
coherence against detection-matched random gene sets; only panels that beat
that null are included. See methodology-notes.md.
"""

import os
import json
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr

# Paths on Stokes.
REPO = "/home/ev039784/gut-gvhd-spatial-pilot"
DATA_ROOT = os.path.join(REPO, "data")
STARFYSH_OUT = os.path.join(REPO, "v2-analysis/outputs/integrated/adata_out.h5ad")
SIGNATURE_CSV = os.path.join(REPO, "v2-analysis/GVHD_spatial_signature.csv")
OUT_DIR = os.path.join(REPO, "v2-analysis/outputs/scores")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = ["SLV11", "SLV12", "SLV13", "SLV14", "SLV15", "SLV16", "SLV17", "SLV18"]

# 5x5 spots of 16um = 80um blocks. Chosen because binning raises counts per
# unit ~21x, which cuts the dropout-driven attenuation that shrinks every
# correlation toward zero. 80um is where the glycolysis module peaks; the
# MHC-II core gains nothing beyond it.
BIN = 5

# Sub-modules are scored separately and then averaged, so each branch of
# glucose handling gets equal weight. A flat mean over all 22 genes would let
# the highest-detection genes dominate and silently reweight the pathways.
GLUCO_MODULES = {
    "Glycolysis": ["HK1", "HK2", "PFKM", "PFKP", "PKM", "GPI", "PGAM1", "ENO1"],
    "PPP":        ["G6PD", "PGLS", "PGD", "TKT", "TALDO1", "RPIA", "RPE"],
    "Pyr_to_TCA": ["PDHA1", "PDHB", "DLAT", "DLD", "PDP1", "MPC1", "MPC2"],
}

PANELS = {
    "OXPHOS": ["CYC1", "UQCRC1", "ATP5MC3", "NDUFV1", "COX4I1",
               "ATP5F1B", "ATP5F1A", "SDHB", "COX7C", "NDUFB2"],
    "FAO": ["CPT1A", "CPT2", "ACADM", "ACADVL", "HADHA",
            "HADHB", "ACAA2", "ACOX1", "ETFA", "ETFB"],
    "TCA": ["MDH2", "IDH2", "FH", "ACO2", "IDH3A", "OGDH"],
    # MHC-I is reported whole and split. The processing arm is interferon-
    # inducible while B2M is closer to constitutive, so averaging them alone
    # would conflate two different quantities.
    "MHC_I":      ["B2M", "TAP1", "TAP2", "PSMB8", "PSMB9", "NLRC5"],
    "MHCI_const": ["B2M"],
    "MHCI_proc":  ["TAP1", "TAP2", "PSMB8", "PSMB9", "NLRC5"],
    # CIITA/RFX only. NFY and CREB1 are ubiquitous transcription factors and
    # measurably dilute this panel. There are no HLA probes in the assay, so
    # this is transactivator expression, not antigen presentation.
    "MHC_II_core": ["CIITA", "RFX5", "RFXANK", "RFXAP"],
    # Kept alone. CD74 diverges from the transactivator core it should track,
    # and has separate biology as the receptor for MIF.
    "CD74": ["CD74"],
    # Control panel: without it, an absent MHC signal cannot be told apart
    # from absent interferon.
    "IFN_gamma": ["STAT1", "IRF1", "GBP1", "GBP2", "CXCL9", "CXCL10"],
}

# Sample metadata, from the reference repo's integration notebook. Half these
# samples are gastric, so the stomach cell types are real tissue here and must
# not be treated as contamination.
META = pd.DataFrame(
    [["SLV11", "C159",  "Antrum",          "Severe"],
     ["SLV12", "C162",  "Rectum",          "Mild"],
     ["SLV13", "C98",   "Stomach_Body",    "Severe"],
     ["SLV14", "C159",  "Rectum",          "Severe"],
     ["SLV15", "C179",  "Antrum",          "Mild"],
     ["SLV16", "C179",  "Ascending_Colon", "Mild"],
     ["SLV17", "ND001", "Ascending_Colon", "ND"],
     ["SLV18", "C162",  "Stomach",         "Mild"]],
    columns=["sample", "patient", "tissue_type", "grade"]).set_index("sample")

GASTRIC = ["Antrum", "Stomach_Body", "Stomach"]

# Cell-type names spelled as in GVHD_spatial_signature.csv.
# Stem is pooled across both tissues, as the reference does
# (stem_cell_types = ['Intestine_Epithelial Stem cell', 'Stomach_stem_cells']).
STEM = ["Intestine Epithelial Stem cells", "Stomach Stem cells"]
EPITHELIUM = STEM + ["Instestinal Epithelial cells",
                     "Stomach Epithelial cells",
                     "Stomach Body Epithelial cells"]
# Excluded everywhere. The Enterocyte signature is small-intestinal (SI, FABP2,
# RBP2, CCL25) and none of these 8 samples are small intestine -- they are
# antrum, stomach, stomach body, rectum, and ascending colon.
ENTEROCYTE = "Enterocyte"


def load_starfysh_proportions(path):
    """Pull per-spot cell-type proportions out of the Starfysh output.

    Read with h5py rather than anndata so we never hold the full expression
    matrix in memory. Checks obsm first, then obs columns.
    """
    with h5py.File(path, "r") as f:
        def decode(arr):
            return np.array([x.decode() if isinstance(x, bytes) else str(x)
                             for x in arr])

        obs = f["obs"]
        idx_key = obs.attrs.get("_index", "_index")
        if isinstance(idx_key, bytes):
            idx_key = idx_key.decode()
        barcodes = decode(obs[idx_key][:])

        # An obsm matrix carries no column labels, so names come from the
        # signature csv -- its column order is what Starfysh was given and the
        # order it returns proportions in.
        sig_names = list(pd.read_csv(SIGNATURE_CSV, nrows=0).columns)

        # Starfysh writes several same-shaped matrices to obsm and only one of
        # them is the deconvolution result. Chosen by name, in priority order,
        # never by pattern match: qc_m is the posterior mean of the cell-type
        # proportions, while qc is a posterior sample, pc_p is the prior, and
        # xs_k is anchor-derived. Picking the wrong one would silently produce
        # different biology rather than an error.
        PREFERRED = ["proportions", "qc_m"]

        if "obsm" in f:
            candidates = {}
            for key in f["obsm"].keys():
                node = f["obsm"][key]
                if not isinstance(node, h5py.Dataset) or node.ndim != 2:
                    continue
                if node.shape[0] != len(barcodes):
                    continue
                if node.shape[1] == len(sig_names):
                    candidates[key] = node

            for key in PREFERRED:
                if key in candidates:
                    mat = np.asarray(candidates[key][:])
                    print(f"  proportions from obsm['{key}'] {mat.shape}, "
                          f"named from {os.path.basename(SIGNATURE_CSV)}")
                    other = sorted(set(candidates) - {key})
                    if other:
                        print(f"  (not used, same shape: {', '.join(other)})")
                    return pd.DataFrame(mat, index=barcodes, columns=sig_names)

            if candidates:
                raise SystemExit(
                    f"\nFound {len(candidates)} matrices of the right shape in "
                    f"obsm -- {', '.join(sorted(candidates))} -- but none named "
                    f"{' or '.join(PREFERRED)}.\n"
                    "These are different quantities (posterior mean vs sample "
                    "vs prior), so picking one by guess would give wrong\n"
                    "proportions without failing. Name the right key explicitly "
                    "in PREFERRED.\n")

        # Fallback: one obs column per cell type, named from the signature csv.
        num_cols = {}
        for key in obs.keys():
            if key.startswith("_"):
                continue
            node = obs[key]
            if isinstance(node, h5py.Dataset) and node.dtype.kind == "f":
                num_cols[key] = node[:]
        hits = [k for k in num_cols if k in EPITHELIUM or k == ENTEROCYTE
                or "cell" in k.lower() or "T cells" in k]
        if hits:
            print(f"  proportions from {len(num_cols)} obs columns")
            return pd.DataFrame(num_cols, index=barcodes)

    raise SystemExit(
        "\nCould not find cell-type proportions in the Starfysh output.\n"
        "v2-starfysh.py assigned model_eval's inference_outputs but only wrote\n"
        "args.adata, so they may never have been attached. Fix by re-running\n"
        "model_eval from the saved model.pt (no retraining needed) and writing\n"
        "inference_outputs['qc_m'] into adata.obsm['proportions'].\n"
        f"Inspect with:\n"
        f"  python -c \"import anndata as ad; a=ad.read_h5ad('{STARFYSH_OUT}');"
        f" print(a); print(list(a.obsm.keys()))\"\n")


def bin_sample(sid):
    """Sum raw counts into 80um blocks for one sample.

    Counts are summed before normalizing. Normalizing per spot first and then
    averaging would throw away exactly the depth the binning is meant to buy.
    """
    with h5py.File(f"{DATA_ROOT}/{sid.lower()}.h5ad", "r") as f:
        def decode(arr):
            return np.array([x.decode() if isinstance(x, bytes) else str(x)
                             for x in arr])
        var = decode(f["var"]["_index"][:])
        obs = decode(f["obs"]["_index"][:])
        X = sp.csr_matrix((f["X"]["data"][:], f["X"]["indices"][:],
                           f["X"]["indptr"][:]), shape=(len(obs), len(var)))

    spatial = f"{DATA_ROOT}/{sid}/binned_outputs/square_016um/spatial"
    pos = pd.read_parquet(f"{spatial}/tissue_positions.parquet")
    pos = pos.set_index("barcode").reindex(obs)

    keep = pos["array_row"].notna().values
    X, pos = X[keep], pos[keep]
    obs = obs[keep]

    row = (pos["array_row"].values // BIN).astype(int)
    col = (pos["array_col"].values // BIN).astype(int)
    block = np.char.add(np.char.add(row.astype(str), "_"), col.astype(str))
    uniq, inv = np.unique(block, return_inverse=True)

    # Sparse indicator matrix; one row per block, one column per spot.
    agg = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))),
                        shape=(len(uniq), len(inv)))
    counts = agg @ X
    depth = np.asarray(counts.sum(1)).ravel()

    block_ids = np.array([f"{sid}_{b}" for b in uniq])
    spot_to_block = pd.Series(block_ids[inv], index=[f"{b}_{sid}" for b in obs])
    return counts.tocsc(), var, depth, block_ids, spot_to_block


def residualize(mat, depth, sample_ids):
    """Regress each gene on log depth, within each sample.

    Removes most of the per-block depth artifact and the between-sample batch
    effect (mean depth varies 7.6x across these samples). It does NOT remove
    all of it: OXPHOS still correlates with depth at ~+0.23 within sample
    afterwards, and a nonparametric version does no better. So this is
    provided alongside the uncorrected scores, not instead of them -- the
    depth adjustment that actually works is a partial correlation at analysis
    time, where you can see what it changes.
    """
    out = mat.copy()
    log_depth = np.log(depth)
    for s in np.unique(sample_ids):
        i = sample_ids == s
        x = log_depth[i] - log_depth[i].mean()
        Y = mat[i]
        Yc = Y - Y.mean(0)
        beta = (x @ Yc) / (x @ x)
        out[i] = Yc - np.outer(x, beta)
    return out


def rank_z(mat):
    """Rank then standardize each gene, so no single gene dominates a panel.

    Average ranks for ties: raw counts are heavily zero-inflated, and breaking
    those ties by array order would impose the spatial ordering of the blocks.
    """
    ranks = np.apply_along_axis(
        lambda c: rankdata(c, method="average"), 0, mat).astype(np.float32)
    sd = ranks.std(0)
    return (ranks - ranks.mean(0)) / np.where(sd == 0, 1, sd)


def score_panels(Z):
    """Panel scores from a rank-standardized expression frame."""
    out = pd.DataFrame(index=Z.index)
    for name, genes in PANELS.items():
        have = [g for g in genes if g in Z.columns]
        if have:
            out[name] = Z[have].mean(1)
    modules = {}
    for name, genes in GLUCO_MODULES.items():
        have = [g for g in genes if g in Z.columns]
        modules[name] = Z[have].mean(1)
        out[name] = modules[name]
    out["GLUCOMETABOLISM"] = pd.DataFrame(modules).mean(1)

    def std(s):
        return (s - s.mean()) / s.std()

    # Glucometabolism and OXPHOS correlate at ~0.57 because both track overall
    # metabolic content, so the raw pair cannot express a fuel preference. The
    # difference of standardized scores drops the shared activity level.
    out["Fuel_contrast"] = std(out["GLUCOMETABOLISM"]) - std(out["OXPHOS"])
    return out


print("Reading Starfysh proportions ...")
props = load_starfysh_proportions(STARFYSH_OUT)
print(f"  {props.shape[0]} spots x {props.shape[1]} cell types")

# Fail here rather than downstream. Without correctly named cell types every
# lookup returns nothing, stem_fraction is never built, and the run still
# produces a plausible-looking parquet with no proportions in it.
found = [c for c in EPITHELIUM if c in props.columns]
if not found:
    raise SystemExit(
        "\nProportions loaded but no expected epithelial cell types are among "
        "the columns.\nExpected some of:\n  "
        + "\n  ".join(EPITHELIUM)
        + f"\nGot {len(props.columns)} columns:\n  "
        + "\n  ".join(map(str, props.columns[:30]))
        + "\n\nThe names must match GVHD_spatial_signature.csv exactly.\n")
print(f"  epithelial types found: {len(found)}/{len(EPITHELIUM)}")

row_sums = props.sum(1)
print(f"  row sums: min={row_sums.min():.3f} median={row_sums.median():.3f} "
      f"max={row_sums.max():.3f}  (~1.0 expected for proportions)")

all_genes = sorted({g for v in PANELS.values() for g in v}
                   | {g for v in GLUCO_MODULES.values() for g in v})

expr_rows, depth_rows, block_rows, sample_rows, prop_rows = [], [], [], [], []

for sid in SAMPLES:
    counts, var, depth, block_ids, spot_to_block = bin_sample(sid)

    present = [g for g in all_genes if g in set(var)]
    missing = sorted(set(all_genes) - set(var))
    idx = [int(np.where(var == g)[0][0]) for g in present]

    # log1p CP10K on block totals.
    sub = np.asarray(counts[:, idx].todense(), dtype=np.float32)
    sub = np.log1p(sub / depth[:, None] * 1e4)

    expr_rows.append(pd.DataFrame(sub, columns=present, index=block_ids))
    depth_rows.append(pd.Series(depth, index=block_ids))
    block_rows.append(block_ids)
    sample_rows.append(np.repeat(sid, len(block_ids)))

    # Average the 16um proportions into the same blocks. Proportions are
    # already per-spot normalized, so the mean is the right aggregation.
    shared = props.index.intersection(spot_to_block.index)
    grouped = props.loc[shared].groupby(spot_to_block.loc[shared]).mean()
    prop_rows.append(grouped.reindex(block_ids))

    print(f"{sid}: {len(spot_to_block)} spots -> {len(block_ids)} blocks, "
          f"median depth {np.median(depth):.0f}, "
          f"{len(shared)} spots matched to Starfysh")
    if missing:
        print(f"  not in probe panel: {missing}")

expr = pd.concat(expr_rows).fillna(0.0)
depth = pd.concat(depth_rows)
samples = np.concatenate(sample_rows)
prop = pd.concat(prop_rows)

print(f"\nTotal: {len(expr)} blocks, median depth {depth.median():.0f}")

# Two versions of every score. The uncorrected one is the input to a partial
# correlation / confound ladder downstream; the "_dadj" one is there to check
# that a conclusion does not hinge on which correction you use. Neither is
# safe to use blind -- see the depth diagnostic printed below.
Z_raw = pd.DataFrame(rank_z(expr.values), index=expr.index,
                     columns=expr.columns)
Z_adj = pd.DataFrame(rank_z(residualize(expr.values, depth.values, samples)),
                     index=expr.index, columns=expr.columns)

scores = score_panels(Z_raw)
scores_adj = score_panels(Z_adj).add_suffix("_dadj")

# Report how much depth is left in each score, estimated within sample so the
# 7.6x between-sample depth spread cannot masquerade as a within-tissue
# gradient. Any panel still well away from zero here cannot be interpreted
# without adjusting for depth in the model.
print("\nResidual depth association, within sample (mean rho across samples):")
print(f"  {'panel':18s} {'uncorrected':>12s} {'_dadj':>10s}")
for name in scores.columns:
    def within(frame, col):
        return np.mean([spearmanr(frame[col][samples == s],
                                  depth.values[samples == s])[0]
                        for s in np.unique(samples)])
    print(f"  {name:18s} {within(scores, name):+12.3f} "
          f"{within(scores_adj, name + '_dadj'):+10.3f}")

out = pd.concat([scores, scores_adj, prop.add_prefix("prop_")], axis=1)
out["sample"] = samples
out["depth"] = depth.values

# Sample-level metadata. grade_group collapses Mild and ND into one arm, the
# same split the reference uses (df_ND['Grade'] = 'mild/no').
for col in ("patient", "tissue_type", "grade"):
    out[col] = META[col].reindex(samples).values
out["grade_group"] = np.where(out["grade"] == "Severe", "severe", "mild_no")
out["tissue_class"] = np.where(out["tissue_type"].isin(GASTRIC),
                               "gastric", "intestinal")

# Stem fraction within epithelium, pooling both stem populations. Using a
# ratio rather than the raw stem proportion sidesteps the compositional
# problem: proportions are constrained to sum to 1, so a raw proportion moves
# whenever any other cell type moves.
stem_cols = [c for c in STEM if c in prop.columns]
epi_cols = [c for c in EPITHELIUM if c in prop.columns]
if stem_cols and epi_cols:
    denom = prop[epi_cols].sum(1)
    out["stem_fraction"] = np.where(denom > 0, prop[stem_cols].sum(1) / denom,
                                    np.nan)
    out["prop_epithelium"] = denom
    missing_ct = sorted(set(EPITHELIUM) - set(prop.columns))
    if missing_ct:
        print(f"WARNING: epithelial types absent from proportions: {missing_ct}")
else:
    print(f"WARNING: no epithelial cell types found in proportions; "
          f"stem_fraction not computed. Columns: {list(prop.columns)}")

t_cols = [c for c in prop.columns if "T cells" in c]
apc_cols = [c for c in prop.columns if c in ("Myeloid", "B cells")]
if t_cols:
    out["prop_Tcell_total"] = prop[t_cols].sum(1)
if apc_cols:
    out["prop_APC_total"] = prop[apc_cols].sum(1)

# Deconvolution sanity check. Gastric samples should be dominated by gastric
# cell types and intestinal samples by intestinal ones. If they are not, the
# proportions are unreliable and nothing downstream of them is interpretable,
# so this is worth reading before anything else.
print("\nDeconvolution check -- top cell types by mean proportion per sample:")
for sid in SAMPLES:
    i = out.index[samples == sid]
    row = prop.loc[i].mean().sort_values(ascending=False)
    top = ", ".join(f"{c} {v:.3f}" for c, v in row.head(3).items())
    sf = out.loc[i, "stem_fraction"].mean() if "stem_fraction" in out else np.nan
    print(f"  {sid} {META.loc[sid, 'tissue_type']:16s} "
          f"{META.loc[sid, 'grade']:7s} stem_frac={sf:.3f}  {top}")

path = os.path.join(OUT_DIR, f"block_scores_{BIN*16}um.parquet")
out.to_parquet(path)
print(f"\nWrote {path}  ({len(out)} rows x {out.shape[1]} cols)")

with open(os.path.join(OUT_DIR, "panel_definitions.json"), "w") as f:
    json.dump({"bin_um": BIN * 16, "panels": PANELS,
               "gluco_modules": GLUCO_MODULES}, f, indent=2)
