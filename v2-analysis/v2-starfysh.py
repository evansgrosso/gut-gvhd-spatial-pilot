#!/usr/bin/env python
# coding: utf-8

# In[10]:


import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import json

from skimage import io
from starfysh import utils

adata_raw = sc.read_h5ad("../v2-analysis/slv14.h5ad")
adata_raw

print(adata_raw)


# In[11]:


adata = adata_raw.copy()

# Remove mitochondrial and ribosomal genes
mt = adata.var_names.str.startswith("MT-")
rb = adata.var_names.str.startswith("RP")

adata = adata[:, ~(mt | rb)].copy()

# Normalize and log-transform
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Identify highly variable genes
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat",
    n_top_genes=2000
)


# In[12]:


adata_raw = adata_raw[
    adata.obs_names,
    adata.var_names
].copy()

adata_raw.var["highly_variable"] = adata.var["highly_variable"]


# In[13]:


gene_sig = pd.read_csv(
    "../v2-analysis/GVHD_spatial_signature.csv"
)

gene_sig.head()

for column in gene_sig.columns:
    gene_sig[column] = gene_sig[column].where(
        gene_sig[column].isin(adata.var_names)
    )

gene_sig


# In[17]:


spatial_dir = "../data/SLV14/binned_outputs/square_016um/spatial"

positions = pd.read_parquet(
    f"{spatial_dir}/tissue_positions.parquet"
)

with open(f"{spatial_dir}/scalefactors_json.json") as f:
    scalefactor = json.load(f)

img = io.imread(
    f"{spatial_dir}/tissue_hires_image.png"
)

positions = positions.set_index("barcode")
positions = positions.loc[adata.obs_names].copy()

positions.head()

map_info = positions[
    [
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres"
    ]
].copy()

map_info.columns = [
    "array_row",
    "array_col",
    "imagecol",
    "imagerow"
]

img_metadata = {
    "map_info": map_info,
    "scalefactor": scalefactor,
    "img": img
}

print(map_info.shape)
print(adata.shape)
print(img.shape)
print(scalefactor)

args = utils.VisiumArguments(
    adata_raw,
    adata,
    gene_sig,
    img_metadata,
    sample_id="SLV14"
)


# In[ ]:


anchors = args.get_anchors()
anchors.head()

anchors.notna().sum()

import inspect


print(inspect.signature(utils.run_starfysh))


# In[21]:


import torch

# Use CPU for compatibility
device = torch.device("cpu")

# Fix Starfysh / SciPy sparse-matrix compatibility
if not isinstance(args.adata.X, np.ndarray):
    args.adata.X = args.adata.X.toarray()

if not isinstance(args.adata_norm.X, np.ndarray):
    args.adata_norm.X = args.adata_norm.X.toarray()

# Run Starfysh
model, losses, adata_out = utils.run_starfysh(
    args,
    n_repeats=3,
    lr=1e-4,
    epochs=100,
    batch_size=32,
    alpha_mul=50,
    poe=False,
    device=device,
    seed=0,
    verbose=True
)


# In[ ]:


print(adata_out)
print(adata_out.obs.columns.tolist())
print(adata_out.obsm.keys())
print(adata_out.uns.keys())


# In[ ]:




