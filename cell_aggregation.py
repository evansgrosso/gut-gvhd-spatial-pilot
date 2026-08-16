import bin2cell as b2c

adata = b2c.read_visium("data/SLV14/binned_outputs/square_002um")

b2c.stardist(
    image_path="data/SLV14/binned_outputs/square_002um/spatial/tissue_lowres_image.png",
    labels_npz_path="stardist_labels.npz",
    stardist_model="2D_versatile_he",
    prob_thresh=0.1
)

b2c.expand_labels(adata, labels_npz_path="stardist_labels.npz", algorithm="volume_ratio")

adata.write("SLV14_cells.h5ad")