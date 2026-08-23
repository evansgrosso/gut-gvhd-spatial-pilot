import anndata as ad
import scanpy as sc
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree


adata_out = ad.read_h5ad("v2-analysis/outputs/SLV14/adata_out.h5ad")

panels = {
    "MHC_II": ["CD74", "CIITA", "RFX5", "RFXAP", "RFXANK"],
    "MHC_I": ["B2M", "TAP1", "TAP2", "NLRC5", "PSMB8", "PSMB9"],
    "Glycolysis": ["SLC2A1", "HK1", "HK2", "PFKP", "PGK1", "ENO1", "PKM"],
    "FAO": ["CPT1A", "CPT2", "ACADM", "ACADVL", "HADHA", "HADHB", "ECHS1", "ACAA2"],
    "Extrinsic_Apoptosis": ["FAS", "FASLG", "FADD", "CASP8", "TNFRSF10A", "TNFRSF10B", "TNFSF10", "CFLAR"],
}

adata_broad = sc.read_h5ad("v2-analysis/slv14.h5ad")
mt = adata_broad.var_names.str.startswith("MT-")
rb = adata_broad.var_names.str.startswith("RP")
adata_broad = adata_broad[:, ~(mt | rb)].copy()
sc.pp.normalize_total(adata_broad)
sc.pp.log1p(adata_broad)

for name, genes in panels.items():
    sc.tl.score_genes(adata_broad, gene_list=genes, score_name=f"{name}_score")

qc_df = pd.DataFrame(adata_out.obsm["qc_m"], columns=adata_out.uns["cell_types"], index=adata_out.obs_names)
scores_df = adata_broad.obs[[f"{name}_score" for name in panels]]

assert set(adata_broad.obs_names) == set(adata_out.obs_names)

combined = qc_df.join(scores_df)
combined["spatial_x"] = adata_out.obsm["spatial"][:, 0]
combined["spatial_y"] = adata_out.obsm["spatial"][:, 1]

t_cell_cols = [c for c in adata_out.uns["cell_types"] if "T cells" in c]
combined["T_cell_total"] = combined[t_cell_cols].sum(axis=1)

isc_prop = combined["Intestine Epithelial Stem cells"].values.reshape(-1, 1)
km_isc = KMeans(n_clusters=2, n_init=10, random_state=0).fit(isc_prop)
isc_cutoff = km_isc.cluster_centers_.mean()
combined["ISC_status"] = np.where(combined["Intestine Epithelial Stem cells"] >= isc_cutoff, "ISC-rich", "ISC-poor")

isc_rich = combined[combined["ISC_status"] == "ISC-rich"].copy()

cd8_cols = [
    "CD8+ Effector T cells", "CD8+ Cytotoxic Unconventional T cells", "CD8+ Proliferating T cells",
    "CD8+ Homeostatic Unconventional T cells", "CD8+ Tissue Resident Memory T cells",
    "CD8+ Transitioning  Resident T cells",
]
combined["CD8_total"] = combined[cd8_cols].sum(axis=1)

mhci_isc = isc_rich["MHC_I_score"].values.reshape(-1, 1)
km_mhci = KMeans(n_clusters=2, n_init=10, random_state=0).fit(mhci_isc)
mhci_cutoff = km_mhci.cluster_centers_.mean()

isc_rich["ISC_MHCI_status"] = np.where(isc_rich["MHC_I_score"] >= mhci_cutoff, "ISC-MHCI rich", "ISC-MHCI poor")
print(isc_rich["ISC_MHCI_status"].value_counts())

cd8_threshold = combined["CD8_total"].quantile(0.75)
cd8_rich_coords = combined.loc[combined["CD8_total"] >= cd8_threshold, ["spatial_x", "spatial_y"]].values
cd8_tree = KDTree(cd8_rich_coords)

mhci_rich_coords = isc_rich.loc[isc_rich["ISC_MHCI_status"] == "ISC-MHCI rich", ["spatial_x", "spatial_y"]].values
mhci_poor_coords = isc_rich.loc[isc_rich["ISC_MHCI_status"] == "ISC-MHCI poor", ["spatial_x", "spatial_y"]].values

nnd_rich_cd8, _ = cd8_tree.query(mhci_rich_coords)
nnd_poor_cd8, _ = cd8_tree.query(mhci_poor_coords)

stat, pvalue = stats.mannwhitneyu(nnd_rich_cd8, nnd_poor_cd8, alternative="two-sided")
mean_rich, mean_poor = nnd_rich_cd8.mean(), nnd_poor_cd8.mean()
std_pooled = np.sqrt((nnd_rich_cd8.var(ddof=1) + nnd_poor_cd8.var(ddof=1)) / 2)
cohens_d = (mean_rich - mean_poor) / std_pooled

print(f"Mean NND to nearest CD8-rich spot — ISC-MHCI rich: {mean_rich:.2f}, poor: {mean_poor:.2f}")
print(f"Mann-Whitney p-value: {pvalue:.4g}")
print(f"Cohen's d: {cohens_d:.3f}")

isc_rich["CD8_total"] = isc_rich[cd8_cols].sum(axis=1)

targets = ["CD8_total", "Glycolysis_score", "FAO_score", "Extrinsic_Apoptosis_score"]
for col in targets:
    r_p, p_p = pearsonr(isc_rich["MHC_I_score"], isc_rich[col])
    r_s, p_s = spearmanr(isc_rich["MHC_I_score"], isc_rich[col])
    print(f"{col}: Pearson r={r_p:.3f} (p={p_p:.3g}), Spearman r={r_s:.3f} (p={p_s:.3g})")


positions = pd.read_parquet("data/SLV14/binned_outputs/square_016um/spatial/tissue_positions.parquet")
positions = positions.set_index("barcode")
positions = positions.loc[combined.index, ["array_row", "array_col"]]
combined["array_row"] = positions["array_row"].values
combined["array_col"] = positions["array_col"].values

cell_type_cols = list(adata_out.uns["cell_types"])
combined["dominant_celltype"] = combined[cell_type_cols].idxmax(axis=1)

stem_types = ["Intestine Epithelial Stem cells", "Stomach Stem cells"]
combined["dominant_celltype_merged"] = combined["dominant_celltype"].replace({t: "stem" for t in stem_types})

stem_coords = combined.loc[combined["dominant_celltype_merged"] == "stem", ["array_row", "array_col"]].values
cd8_coords = combined.loc[combined["dominant_celltype"] == "CD8+ Effector T cells", ["array_row", "array_col"]].values
print("stem-dominant spots:", len(stem_coords))
print("CD8-effector-dominant spots:", len(cd8_coords))

cd8_tree_grid = KDTree(cd8_coords)
nnd_stem_to_cd8, _ = cd8_tree_grid.query(stem_coords)
print("Mean NND (stem -> nearest CD8 effector), grid units:", nnd_stem_to_cd8.mean())
print("Median NND:", np.median(nnd_stem_to_cd8))