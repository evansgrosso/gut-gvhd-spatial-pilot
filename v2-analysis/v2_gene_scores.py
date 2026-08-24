import anndata as ad
import scanpy as sc
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.cluster import KMeans


PANELS = {
    "MHC_II": ["CD74", "CIITA", "RFX5", "RFXAP", "RFXANK"],
    "MHC_I": ["B2M", "TAP1", "TAP2", "NLRC5", "PSMB8", "PSMB9"],
    "Glycolysis": ["SLC2A1", "HK1", "HK2", "PFKP", "PGK1", "ENO1", "PKM"],
    "FAO": ["CPT1A", "CPT2", "ACADM", "ACADVL", "HADHA", "HADHB", "ECHS1", "ACAA2"],
    "Extrinsic_Apoptosis": ["FAS", "FASLG", "FADD", "CASP8", "TNFRSF10A", "TNFRSF10B", "TNFSF10", "CFLAR"],
}

SCALE_FACTOR = 0.5097273  # tissue_hires_scalef

# Region boundaries taken directly from the reference paper's own hand-drawn
# bounding boxes for SLV14 (CellphoneDB_Analysis.ipynb), in hires-image pixel
# space. Each box is (y_min, y_max, x_min, x_max).
CRYPT_LOSS_BOXES = [
    (2300, 2900, 1000, 1400),
    (1800, 2000, 1600, 2100),
]
CRYPT_INTACT_BOXES = [
    (1400, 1800, 1700, 2300),
]


def _in_boxes(x, y, boxes):
    mask = pd.Series(False, index=x.index)
    for y0, y1, x0, x1 in boxes:
        mask |= (y >= y0) & (y <= y1) & (x >= x0) & (x <= x1)
    return mask


def load_combined():
    """Load Starfysh output + signature scores into one per-spot DataFrame.

    Returns (combined, cell_types) where combined has: Starfysh cell-type
    proportions, `{name}_score` for each PANELS entry, spatial_x/spatial_y,
    img_x/img_y (hires-image pixel coords), crypt_loss/crypt_intact (bool),
    and ISC_status (ISC-rich/ISC-poor via k-means).
    """
    adata_out = ad.read_h5ad("v2-analysis/outputs/SLV14/adata_out.h5ad")

    adata_broad = sc.read_h5ad("v2-analysis/slv14.h5ad")
    mt = adata_broad.var_names.str.startswith("MT-")
    rb = adata_broad.var_names.str.startswith("RP")
    adata_broad = adata_broad[:, ~(mt | rb)].copy()
    sc.pp.normalize_total(adata_broad)
    sc.pp.log1p(adata_broad)

    for name, genes in PANELS.items():
        sc.tl.score_genes(adata_broad, gene_list=genes, score_name=f"{name}_score")

    assert set(adata_broad.obs_names) == set(adata_out.obs_names)

    cell_types = adata_out.uns["cell_types"]
    qc_df = pd.DataFrame(adata_out.obsm["qc_m"], columns=cell_types, index=adata_out.obs_names)
    scores_df = adata_broad.obs[[f"{name}_score" for name in PANELS]]

    combined = qc_df.join(scores_df)
    combined["spatial_x"] = adata_out.obsm["spatial"][:, 0]
    combined["spatial_y"] = adata_out.obsm["spatial"][:, 1]
    combined["img_x"] = combined["spatial_x"] * SCALE_FACTOR
    combined["img_y"] = combined["spatial_y"] * SCALE_FACTOR

    combined["crypt_loss"] = _in_boxes(combined["img_x"], combined["img_y"], CRYPT_LOSS_BOXES)
    combined["crypt_intact"] = _in_boxes(combined["img_x"], combined["img_y"], CRYPT_INTACT_BOXES)

    isc_prop = combined["Intestine Epithelial Stem cells"].values.reshape(-1, 1)
    km_isc = KMeans(n_clusters=2, n_init=10, random_state=0).fit(isc_prop)
    isc_cutoff = km_isc.cluster_centers_.mean()
    combined["ISC_status"] = np.where(combined["Intestine Epithelial Stem cells"] >= isc_cutoff, "ISC-rich", "ISC-poor")

    return combined, cell_types


if __name__ == "__main__":
    combined, cell_types = load_combined()

    print("Crypt-loss spots:", combined["crypt_loss"].sum())
    print("Crypt-intact spots:", combined["crypt_intact"].sum())

    for col in ["MHC_II_score", "MHC_I_score"]:
        loss_vals = combined.loc[combined["crypt_loss"], col]
        intact_vals = combined.loc[combined["crypt_intact"], col]
        stat, pvalue = stats.mannwhitneyu(loss_vals, intact_vals, alternative="two-sided")
        print(f"\n{col}")
        print(f"  crypt-loss mean:   {loss_vals.mean():.4f}")
        print(f"  crypt-intact mean: {intact_vals.mean():.4f}")
        print(f"  Mann-Whitney p:    {pvalue:.4g}")

    isc_rich = combined[combined["ISC_status"] == "ISC-rich"]

    print("ISC-rich spots in crypt-loss:", isc_rich["crypt_loss"].sum())
    print("ISC-rich spots in crypt-intact:", isc_rich["crypt_intact"].sum())

    for col in ["MHC_II_score", "MHC_I_score"]:
        loss_vals = isc_rich.loc[isc_rich["crypt_loss"], col]
        intact_vals = isc_rich.loc[isc_rich["crypt_intact"], col]
        stat, pvalue = stats.mannwhitneyu(loss_vals, intact_vals, alternative="two-sided")
        print(f"\n{col} (ISC-rich only)")
        print(f"  crypt-loss mean:   {loss_vals.mean():.4f}")
        print(f"  crypt-intact mean: {intact_vals.mean():.4f}")
        print(f"  Mann-Whitney p:    {pvalue:.4g}")
