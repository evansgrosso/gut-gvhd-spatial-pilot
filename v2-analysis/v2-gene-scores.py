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