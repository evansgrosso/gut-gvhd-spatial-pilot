import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from pathlib import Path

from v2_gene_scores import load_combined, PANELS, CRYPT_LOSS_BOXES, CRYPT_INTACT_BOXES

TISSUE_IMAGE = "data/SLV14/binned_outputs/square_016um/spatial/tissue_hires_image.png"
FIGURES_DIR = Path("v2-analysis/outputs/SLV14/figures")


def plot_spatial_signatures(combined, tissue_img, signature_names=None, save_path=None):
    """One panel per signature: tissue image with spots colored by score."""
    if signature_names is None:
        signature_names = list(PANELS)

    fig, axes = plt.subplots(1, len(signature_names), figsize=(4.5 * len(signature_names), 4.5))
    if len(signature_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, signature_names):
        col = f"{name}_score"
        vmin, vmax = combined[col].quantile([0.02, 0.98])
        ax.imshow(tissue_img)
        sca = ax.scatter(
            combined["img_x"], combined["img_y"], c=combined[col],
            cmap="viridis", s=2, alpha=0.6, linewidths=0, vmin=vmin, vmax=vmax,
        )
        ax.set_title(name.replace("_", " "))
        ax.axis("off")
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if save_path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    return fig


def plot_region_boxes(tissue_img, save_path=None):
    """Tissue image with the crypt-loss / crypt-intact bounding boxes drawn on it."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(tissue_img)

    for y0, y1, x0, x1 in CRYPT_LOSS_BOXES:
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.5))
    for y0, y1, x0, x1 in CRYPT_INTACT_BOXES:
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="blue", linewidth=1.5))

    ax.legend(
        handles=[
            mpatches.Patch(edgecolor="red", facecolor="none", label="Crypt-loss"),
            mpatches.Patch(edgecolor="blue", facecolor="none", label="Crypt-intact"),
        ],
        loc="upper right",
    )
    ax.set_title("Crypt-loss / crypt-intact regions (SLV14)")
    ax.axis("off")

    fig.tight_layout()
    if save_path:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    return fig


if __name__ == "__main__":
    combined, cell_types = load_combined()
    tissue_img = mpimg.imread(TISSUE_IMAGE)
    plot_spatial_signatures(combined, tissue_img, save_path=FIGURES_DIR / "spatial_signatures.png")
    plot_region_boxes(tissue_img, save_path=FIGURES_DIR / "region_boxes.png")
