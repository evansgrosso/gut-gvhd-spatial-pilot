
import os
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import json
from types import SimpleNamespace
from skimage import io
from starfysh import utils, utils_integrate
from starfysh.starfysh import model_eval
import inspect

# Define configuration constants: sample list, data root, output directory.
SAMPLES = ["SLV11", "SLV12", "SLV13", "SLV14", "SLV15", "SLV16", "SLV17", "SLV18"]
DATA_ROOT = "data"
OUT_DIR = os.path.join("v2-analysis", "outputs", "integrated")
os.makedirs(OUT_DIR, exist_ok=True)

# Load spatial signature csv file.
gene_sig = pd.read_csv("v2-analysis/GVHD_spatial_signature.csv")

# Define variables to store intermediary data
raw_list, norm_list, img_metadata = [], [], {}

print(f"Loading {len(SAMPLES)} samples...")
for sid in SAMPLES:
    # Read the h5ad file produced by QC script
    a = sc.read_h5ad(f"{DATA_ROOT}/{sid.lower()}.h5ad")
    
    # Make barcodes unique by appending sample ID
    a.obs_names = [f"{bc}_{sid}" for bc in a.obs_names]
    
    # Tags which sample each spot came from.
    a.obs["sample"] = sid
    
    # Remove mitochondrial and ribosomal genes.
    mt = a.var_names.str.startswith(("MT-", "mt-"))
    rb = a.var_names.str.startswith(("RP-", "rp-"))
    a = a[:, ~(mt | rb)].copy()
    
    # Create normalized copy for HVG selection
    n = a.copy()
    sc.pp.normalize_total(n)
    sc.pp.log1p(n)
    
    # Store raw and normalized versions
    raw_list.append(a)
    norm_list.append(n)
    
    # Load spatial metadata and attach it for this sample
    spatial_dir = f"{DATA_ROOT}/{sid}/binned_outputs/square_016um/spatial"
    
    # Read tissue positions
    positions = pd.read_parquet(f"{spatial_dir}/tissue_positions.parquet")
    
    # Update barcodes in positions
    positions["barcode"] = positions["barcode"].astype(str) + f"_{sid}"
    
    # Subset to only spots in the current sample
    positions = positions.set_index("barcode").loc[a.obs_names].copy()
    
    # Extract and rename spatial coordinates to match Starfysh's expected format
    map_info = positions[["array_row", "array_col",
                          "pxl_col_in_fullres", "pxl_row_in_fullres"]].copy()
    map_info.columns = ["array_row", "array_col", "imagecol", "imagerow"]
    
    # Load scale factors
    with open(f"{spatial_dir}/scalefactors_json.json") as f:
        scalefactor = json.load(f)
    
    # Load H&E image
    img = io.imread(f"{spatial_dir}/tissue_hires_image.png")
    
    # Store all metadata for this sample, keyed by sample ID
    img_metadata[sid] = {
        "map_info": map_info,
        "scalefactor": scalefactor,
        "img": img,
    }

print("Concatenating samples...")
# Concatenate all raw data vertically (union of genes across samples)
adata_raw_all = ad.concat(raw_list, axis=0, join="inner")
adata_norm_all = ad.concat(norm_list, axis=0, join="inner")
del raw_list, norm_list  # Free RAM

# Select HVGs accounting for differences in each batch.
sc.pp.highly_variable_genes(adata_norm_all)

# Flag raw data with HVG tags
adata_raw_all.var["highly_variable"] = adata_norm_all.var["highly_variable"].values
print("Selected HVGs.")

# Filter gene signatures to only those included in the full gene panel
for col in gene_sig.columns:
    gene_sig[col] = gene_sig[col].where(gene_sig[col].isin(adata_norm_all.var_names))
    
# Split data back to per-sample.
per_raw = {s: adata_raw_all[adata_raw_all.obs["sample"] == s].copy() for s in SAMPLES}
per_norm = {s: adata_norm_all[adata_norm_all.obs["sample"] == s].copy() for s in SAMPLES}

print("Building per-sample signature scores...")
# STEP 4: Build per-sample VisiumArguments (within-sample signature scoring)
individual_args = {}
for sid in SAMPLES:
    va = utils.VisiumArguments(
        per_raw[sid],
        per_norm[sid],
        gene_sig,
        img_metadata[sid],
        sample_id=sid,
        window_size=1,
        patch_r=13,
    )

    # Extract the signature scores (the integrate wrapper concatenates these)
    individual_args[sid] = SimpleNamespace(
        sig_mean=va.sig_mean,
        sig_mean_norm=va.sig_mean_norm
    )
    del va  # Free RAM

print("Building integrated Starfysh arguments...")
# Defines Starfysh arguments
args = utils_integrate.VisiumArguments_integrate(
    adata_raw_all,
    adata_norm_all,
    gene_sig,
    img_metadata,
    individual_args,
    sample_id=SAMPLES,
    window_size=1,
    patch_r=13,
)

# Convert to dense matrix
args.adata.X = args.adata.X.toarray()

print("Starting Starfysh training...")
# Run Starfysh training
model, loss = utils_integrate.run_starfysh(
    args,
    n_repeats=3,        # Paper's value: 3 independent training runs
    lr=1e-4,            # Paper's value: learning rate
    epochs=100,         # Paper's value: max epochs per run
    batch_size=32,      # Paper's value: batch size
    alpha_mul=50,       # Paper's value: annealing schedule multiplier
    poe=False,          # Paper's value: no image integration (PoE) for this dataset
    device= "cuda",
    verbose=True,       # Print progress
)
print("Training complete.")

print("Running model evaluation...")
# Extract deconvoluted spot proportions
inference_outputs, generative_outputs = model_eval(
    model,
    args.adata,
    args,
    poe=False,
    device="cuda",
)

adata_out = args.adata

print("Saving outputs...")
# Save the annotated data with proportions
adata_out.write(os.path.join(OUT_DIR, "adata_out.h5ad"))

# Save the trained model weights
torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))

# Save training losses for diagnostics
with open(os.path.join(OUT_DIR, "losses.json"), "w") as f:
    json.dump(loss, f)

print("Done.")