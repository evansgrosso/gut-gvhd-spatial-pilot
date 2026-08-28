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
ENTEROCYTE = "Enterocyte"

# Sub-modules are scored separately and then averaged, so each branch of
# glucose handling (breakdown, the pentose shunt, entry into the TCA cycle)
# gets equal say -- a flat mean over all 22 genes would let whichever branch
# has the most highly-detected genes dominate.
GLUCO_MODULES = {
    "Glycolysis": ["HK1", "HK2", "PFKM", "PFKP", "PKM", "GPI", "PGAM1", "ENO1"],
    "PPP":        ["G6PD", "PGLS", "PGD", "TKT", "TALDO1", "RPIA", "RPE"],
    "Pyr_to_TCA": ["PDHA1", "PDHB", "DLAT", "DLD", "PDP1", "MPC1", "MPC2"],
}
OXPHOS_MODULES = {

}

# Defines gene panels for desired cell expression
PANELS = {
    "MHC_I":      ["B2M", "TAP1", "TAP2", "PSMB8", "PSMB9", "NLRC5"],
    "MHC_II_core": ["CIITA", "RFX5", "RFXANK", "RFXAP"],
    "CD74": ["CD74"],
    "IFN_gamma": ["STAT1", "IRF1", "GBP1", "GBP2", "CXCL9", "CXCL10"],
}

# Loads proportions as computed by Starfysh
def load_starfysh_proportions(path):
    sig_names = list(pd.read_csv(SIGNATURE_CSV, nrows=0).columns)
    PREFERRED = ["proportions", "qc_m"]

    with h5py.File(path, "r") as f:
        def decode(arr):
            return np.array([x.decode() if isinstance(x, bytes) else str(x)
                             for x in arr])

        obs = f["obs"]
        idx_key = obs.attrs.get("_index", "_index")
        if isinstance(idx_key, bytes):
            idx_key = idx_key.decode()
        barcodes = decode(obs[idx_key][:])

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

        # Fallback: one obs column per cell type, named directly.
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
        f"Inspect with:\n"
        f"  python -c \"import h5py; f=h5py.File('{STARFYSH_OUT}','r'); "
        f"print(list(f['obsm'].keys()))\"\n")

# Sums a sample's raw counts into 80 micrometer blocks. Normalizes and log-transforms the counts.
def bin_and_normalize(sid, genes_wanted):
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
    block_ids, inv = np.unique(block, return_inverse=True)

    agg = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))),
                        shape=(len(block_ids), len(inv)))
    counts = agg @ X
    depth = np.asarray(counts.sum(1)).ravel()

    present = [g for g in genes_wanted if g in set(var)]
    missing = sorted(set(genes_wanted) - set(var))
    idx = [int(np.where(var == g)[0][0]) for g in present]

    raw = np.asarray(counts[:, idx].todense(), dtype=np.float64)
    cpm = raw / depth[:, None] * 1e4                    # step 2
    log_cpm = np.log1p(cpm)                             # step 3
    log_cpm = pd.DataFrame(log_cpm, columns=present, index=block_ids)

    spot_to_block = pd.Series(block_ids[inv],
                              index=[f"{b}_{sid}" for b in obs])
    return log_cpm, depth, block_ids, spot_to_block, missing

# Compute z-score of each gene across the sample's blocks. 
def score_panel(log_cpm, genes):
    """Steps 4-5: z-score each gene across this sample's blocks, then
    average across the panel's genes.
    """
    have = [g for g in genes if g in log_cpm.columns]
    if not have:
        return None
    z = (log_cpm[have] - log_cpm[have].mean()) / log_cpm[have].std()
    return z.mean(axis=1)


print("Reading Starfysh proportions ...")
props = load_starfysh_proportions(STARFYSH_OUT)
print(f"  {props.shape[0]} spots x {props.shape[1]} cell types")

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

score_rows, prop_rows, depth_rows, sample_rows, block_index = [], [], [], [], []

for sid in SAMPLES:
    log_cpm, depth, block_ids, spot_to_block, missing = bin_and_normalize(
        sid, all_genes)

    scores = pd.DataFrame(index=block_ids)
    for name, genes in PANELS.items():
        s = score_panel(log_cpm, genes)
        if s is not None:
            scores[name] = s.values

    modules = {}
    for name, genes in GLUCO_MODULES.items():
        s = score_panel(log_cpm, genes)
        modules[name] = s
        scores[name] = s.values
    scores["GLUCOMETABOLISM"] = pd.DataFrame(modules).mean(axis=1).values

    scores["GLUCO - OXPHOS"] = scores["GLUCOMETABOLISM"] - scores["OXPHOS"]

    # Average the 16um proportions into the same blocks. 
    shared = props.index.intersection(spot_to_block.index)
    grouped = props.loc[shared].groupby(spot_to_block.loc[shared]).mean()

    score_rows.append(scores)
    prop_rows.append(grouped.reindex(block_ids))
    depth_rows.append(pd.Series(depth, index=block_ids))
    sample_rows.append(np.repeat(sid, len(block_ids)))
    block_index.append(block_ids)

scores = pd.concat(score_rows, ignore_index=False)
scores.index = np.concatenate([[f"{s}_{b}" for b in ids]
                               for s, ids in zip(SAMPLES, block_index)])
prop = pd.concat(prop_rows)
prop.index = scores.index
depth = pd.concat(depth_rows)
depth.index = scores.index
samples = np.concatenate(sample_rows)

print(f"\nTotal: {len(scores)} blocks, median depth {depth.median():.0f}")

out = pd.concat([scores, prop.add_prefix("prop_")], axis=1)
out["sample"] = samples
out["depth"] = depth.values

for col in ("patient", "tissue_type", "grade"):
    out[col] = META[col].reindex(samples).values
# Collapses Mild and ND into one arm, the same split the reference paper
# uses (its own code: df_ND['Grade'] = 'mild/no').
out["grade_group"] = np.where(out["grade"] == "Severe", "severe", "mild_no")
out["tissue_class"] = np.where(out["tissue_type"].isin(GASTRIC),
                               "gastric", "intestinal")

# STEM cell fraction within epithelium
stem_cols = [c for c in STEM if c in prop.columns]
epi_cols = [c for c in EPITHELIUM if c in prop.columns]
denom = prop[epi_cols].sum(1)
out["stem_fraction"] = np.where(denom > 0, prop[stem_cols].sum(1) / denom, np.nan)
out["prop_epithelium"] = denom

# T cell and APC proportions
t_cols = [c for c in prop.columns if "T cells" in c]
apc_cols = [c for c in prop.columns if c in ("Myeloid", "B cells")]
if t_cols:
    out["prop_Tcell_total"] = prop[t_cols].sum(1)
if apc_cols:
    out["prop_APC_total"] = prop[apc_cols].sum(1)


path = os.path.join(OUT_DIR, f"block_scores_{BIN*16}um.parquet")
out.to_parquet(path)
print(f"\nWrote {path}  ({len(out)} rows x {out.shape[1]} cols)")

with open(os.path.join(OUT_DIR, "panel_definitions.json"), "w") as f:
    json.dump({"bin_um": BIN * 16, "panels": PANELS,
               "gluco_modules": GLUCO_MODULES}, f, indent=2)
