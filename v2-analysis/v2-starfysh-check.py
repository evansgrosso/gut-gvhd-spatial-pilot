import anndata as ad
import numpy as np
import json

adata = ad.read_h5ad("v2-analysis/outputs/SLV14/adata_out.h5ad")
with open("v2-analysis/outputs/SLV14/losses.json") as f:
    losses = json.load(f)

print(adata.shape)                    # expect (28038, ~2152)
qc = adata.obsm["qc_m"]
row_sums = qc.sum(axis=1)
print(row_sums.min(), row_sums.max()) # should be ~1.0 both ends
print(np.isnan(qc).sum())             # should be 0

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, key in zip(axes.flat, ["tot", "reconst", "u", "z", "c", "n"]):
    ax.plot(losses[key])
    ax.set_title(key)
plt.tight_layout()
plt.show()

import pandas as pd

cell_types = list(adata.uns["cell_types"])
qc_df = pd.DataFrame(adata.obsm["qc_m"], columns=cell_types, index=adata.obs_names)

t_cell_cols = [c for c in cell_types if "T cells" in c]
print(t_cell_cols)
qc_df["T_cell_total"] = qc_df[t_cell_cols].sum(axis=1)

coords = adata.obsm["spatial"]
x, y = coords[:, 0], coords[:, 1]

fig, ax = plt.subplots(figsize=(8, 8))
s = ax.scatter(x, y, c=qc_df["T_cell_total"].values, s=1, cmap="viridis")
ax.set_title("Total T cells (all subsets summed)")
ax.invert_yaxis()
ax.set_aspect("equal")
plt.colorbar(s, ax=ax)
plt.tight_layout()
plt.show()