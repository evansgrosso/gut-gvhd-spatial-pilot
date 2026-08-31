import os
import json
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = "/home/ev039784/gut-gvhd-spatial-pilot"
DATA_ROOT = os.path.join(REPO, "data")
STARFYSH_OUT = os.path.join(REPO, "v2-analysis/outputs/integrated/adata_out.h5ad")
SIGNATURE_CSV = os.path.join(REPO, "v2-analysis/GVHD_spatial_signature.csv")
OUT_DIR = os.path.join(REPO, "v2-analysis/outputs/scores")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = ["SLV11", "SLV12", "SLV13", "SLV14", "SLV15", "SLV16", "SLV17", "SLV18"]
BIN = 5 # Scaling factor for bin multiplier. 16x16 micrometers to 80x80 micrometers

# Sample metadata, obtained from the reference repository
META = pd.DataFrame(
    [["SLV11", "C159", "Antrum", "Severe"],
     ["SLV12", "C162", "Rectum", "Mild"],
     ["SLV13", "C98", "Stomach_Body", "Severe"],
     ["SLV14", "C159", "Rectum", "Severe"],
     ["SLV15", "C179", "Antrum", "Mild"],
     ["SLV16", "C179", "Ascending_Colon", "Mild"],
     ["SLV17", "ND001", "Ascending_Colon", "ND"],
     ["SLV18", "C162", "Stomach", "Mild"]],
    columns=["sample", "patient", "tissue_type", "grade"]).set_index("sample")
GASTRIC = ["Antrum", "Stomach_Body", "Stomach"] # Subgroup to contain gastric samples.

# Subgroup cell signatures into desired groups. STEM cells and mature epithelial cells.
STEM = ["Intestine Epithelial Stem cells", "Stomach Stem cells"] # Sums each type of stem cell (instestinal vs stomach) under one group.
EPITHELIUM = STEM + ["Instestinal Epithelial cells", "Stomach Epithelial cells", "Stomach Body Epithelial cells"] # Sums each type of epithelial cell under one group

# Sub-modules are scored separately and then averaged.
GLUCO_MODULES = {
    "Glycolysis": ["HK1", "HK2", "PFKM", "PFKP", "PKM", "GPI", "PGAM1", "ENO1"],
    "PPP":        ["G6PD", "PGLS", "PGD", "TKT", "TALDO1", "RPIA", "RPE"],
    "Pyr_to_TCA": ["PDHA1", "PDHB", "DLAT", "DLD", "PDP1", "MPC1", "MPC2"],
}

# Sub-modules are scored separately and then averaged.
OXPHOS_MODULES = {
    "ETC_CI":   ["NDUFA1", "NDUFA2", "NDUFA3", "NDUFA8", "NDUFA9", "NDUFA10",
                 "NDUFA11", "NDUFA13", "NDUFAB1", "NDUFB1", "NDUFB2", "NDUFB3",
                 "NDUFB5", "NDUFB6", "NDUFB7", "NDUFB8", "NDUFB9", "NDUFB10",
                 "NDUFB11", "NDUFC1", "NDUFC2", "NDUFS1", "NDUFS2", "NDUFS3",
                 "NDUFS4", "NDUFS5", "NDUFS6", "NDUFS7", "NDUFV1", "NDUFV2",
                 "NDUFV3"],
    "ETC_CIII": ["UQCRC1", "UQCRC2", "UQCRB", "UQCRQ", "UQCRH", "UQCR10",
                 "UQCR11", "CYC1"],
    "ETC_CIV":  ["COX4I1", "COX6A1", "COX6B1", "COX6C", "COX7A2", "COX7B",
                 "COX7C", "COX8A"],
    "ETC_CV":   ["ATP5F1A", "ATP5F1B", "ATP5F1C", "ATP5F1D", "ATP5F1E",
                 "ATP5MC1", "ATP5MC2", "ATP5MC3", "ATP5ME", "ATP5MF", "ATP5MG",
                 "ATP5PB", "ATP5PD", "ATP5PF"],
    "TCA":      ["ACO2", "IDH2", "IDH3A", "IDH3B", "IDH3G", "OGDH", "SUCLG1",
                 "SUCLG2", "SDHA", "SDHB", "FH", "MDH2"],
}

# Defines gene panels for desired cell expression
PANELS = {
    "MHC_I":      ["B2M", "TAP1", "TAP2", "PSMB8", "PSMB9", "NLRC5"],
    "MHC_II_core": ["CIITA", "RFX5", "RFXANK", "RFXAP"],
    "CD74": ["CD74"],
    "IFN_gamma": ["STAT1", "IRF1", "GBP1", "GBP2", "CXCL9", "CXCL10"],
}

# Decodes h5py byte strings into a string array
def decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])

# Loads proportions as computed by Starfysh
def load_starfysh_proportions(path):
    sig_names = list(pd.read_csv(SIGNATURE_CSV, nrows=0).columns)
    with h5py.File(path, "r") as f:
        barcodes = decode(f["obs"]["_index"][:])
        mat = f["obsm"]["qc_m"][:]
    return pd.DataFrame(mat, index=barcodes, columns=sig_names)

# Sums a sample's raw counts into 80 micrometer blocks. Normalizes and log-transforms the counts.
def bin_and_normalize(sid, genes_wanted):
    with h5py.File(f"{DATA_ROOT}/{sid.lower()}.h5ad", "r") as f:
        var = decode(f["var"]["_index"][:])
        obs = decode(f["obs"]["_index"][:])
        X = sp.csr_matrix((f["X"]["data"][:], f["X"]["indices"][:],
                           f["X"]["indptr"][:]), shape=(len(obs), len(var)))

    spatial = f"{DATA_ROOT}/{sid}/binned_outputs/square_016um/spatial"
    pos = pd.read_parquet(f"{spatial}/tissue_positions.parquet")
    pos = pos.set_index("barcode").reindex(obs)

    keep = pos["array_row"].notna().values
    X, pos, obs = X[keep], pos[keep], obs[keep]

    row = (pos["array_row"].values // BIN).astype(int)
    col = (pos["array_col"].values // BIN).astype(int)
    block = np.char.add(np.char.add(row.astype(str), "_"), col.astype(str))
    block_ids, inv = np.unique(block, return_inverse=True)

    agg = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))),
                        shape=(len(block_ids), len(inv)))
    counts = agg @ X
    depth = np.asarray(counts.sum(1)).ravel()

    vidx = {g: i for i, g in enumerate(var)}
    present = [g for g in genes_wanted if g in vidx]
    raw = np.asarray(counts[:, [vidx[g] for g in present]].todense(), dtype=np.float64)
    log_cpm = pd.DataFrame(np.log1p(raw / depth[:, None] * 1e4),
                           columns=present, index=block_ids)

    spot_to_block = pd.Series(block_ids[inv], index=[f"{b}_{sid}" for b in obs])
    return log_cpm, depth, block_ids, spot_to_block

# Compute z-score of each gene across the sample's blocks.
def score_panel(log_cpm, genes):
    have = [g for g in genes if g in log_cpm.columns]
    z = (log_cpm[have] - log_cpm[have].mean()) / log_cpm[have].std()
    return z.mean(axis=1)

# Scores each sub-module on its own, then averages them into the group total.
def score_modules(log_cpm, scores, modules, total):
    mods = {}
    for name, genes in modules.items():
        mods[name] = score_panel(log_cpm, genes)
        scores[name] = mods[name].values
    scores[total] = pd.DataFrame(mods).mean(axis=1).values

props = load_starfysh_proportions(STARFYSH_OUT)

all_genes = sorted({g for v in PANELS.values() for g in v}
                   | {g for v in GLUCO_MODULES.values() for g in v}
                   | {g for v in OXPHOS_MODULES.values() for g in v})

score_rows, prop_rows = [], []

for sid in SAMPLES:
    log_cpm, depth, block_ids, spot_to_block = bin_and_normalize(sid, all_genes)

    scores = pd.DataFrame(index=[f"{sid}_{b}" for b in block_ids])
    for name, genes in PANELS.items():
        scores[name] = score_panel(log_cpm, genes).values
    score_modules(log_cpm, scores, GLUCO_MODULES, "GLUCOMETABOLISM")
    score_modules(log_cpm, scores, OXPHOS_MODULES, "OXPHOS")

    # Positive means glucose-favored. Pyruvate entry counts as the last step of
    # glucose handling, so the split falls at the TCA cycle itself.
    scores["GLUCO - OXPHOS"] = scores["GLUCOMETABOLISM"] - scores["OXPHOS"]

    # Average the 16um proportions into the same blocks.
    shared = props.index.intersection(spot_to_block.index)
    grouped = props.loc[shared].groupby(spot_to_block.loc[shared]).mean()
    grouped = grouped.reindex(block_ids).add_prefix("prop_")
    grouped.index = scores.index
    grouped["sample"] = sid
    grouped["depth"] = depth

    score_rows.append(scores)
    prop_rows.append(grouped)

out = pd.concat([pd.concat(score_rows), pd.concat(prop_rows)], axis=1)

for col in ("patient", "tissue_type", "grade"):
    out[col] = META[col].reindex(out["sample"]).values
out["grade_group"] = np.where(out["grade"] == "Severe", "severe", "mild_no")
out["tissue_class"] = np.where(out["tissue_type"].isin(GASTRIC),
                               "gastric", "intestinal")

# STEM cell fraction within epithelium
denom = out[[f"prop_{c}" for c in EPITHELIUM]].sum(1)
out["stem_fraction"] = np.where(
    denom > 0, out[[f"prop_{c}" for c in STEM]].sum(1) / denom, np.nan)
out["prop_epithelium"] = denom

# T cell and APC proportions
out["prop_Tcell_total"] = out[[c for c in out.columns if "T cells" in c]].sum(1)
out["prop_APC_total"] = out[["prop_Myeloid", "prop_B cells"]].sum(1)

out.to_parquet(os.path.join(OUT_DIR, f"block_scores_{BIN*16}um.parquet"))

with open(os.path.join(OUT_DIR, "panel_definitions.json"), "w") as f:
    json.dump({"bin_um": BIN * 16, "panels": PANELS,
               "gluco_modules": GLUCO_MODULES,
               "oxphos_modules": OXPHOS_MODULES}, f, indent=2)
