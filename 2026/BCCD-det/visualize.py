import argparse
import json
import os
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Color scheme matching meta.json or high-contrast readable colors
CLASS_COLORS = {
    "RBC": "#D0021B",       # Red
    "WBC": "#4A90E2",       # Blue
    "Platelets": "#7ED321", # Green
}
DEFAULT_COLOR = "#F5A623"   # Orange fallback


def load_annotations(ann_path):
    """Load Supervisely format JSON annotation."""
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_ground_truth(image_path, ann_path, save_path=None, show=True):
    """Plot image with ground truth bounding boxes and class labels."""
    img = Image.open(image_path).convert("RGB")
    ann = load_annotations(ann_path)

    fig, ax = plt.subplots(1, figsize=(10, 7.5))
    ax.imshow(img)

    objects = ann.get("objects", [])
    class_counts = {}

    for obj in objects:
        class_name = obj.get("classTitle", "Unknown")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        points = obj.get("points", {}).get("exterior", [])
        if len(points) == 2:
            (xmin, ymin), (xmax, ymax) = points
            width = xmax - xmin
            height = ymax - ymin

            # Draw bounding box
            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            # Draw label banner
            ax.text(
                xmin,
                max(0, ymin - 3),
                f"{class_name}",
                color="white",
                fontsize=8,
                fontweight="bold",
                bbox=dict(
                    facecolor=color,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                    alpha=0.85,
                ),
            )

    # Title with class counts
    counts_str = ", ".join([f"{k}: {v}" for k, v in class_counts.items()])
    title = f"{Path(image_path).name} | Total Objects: {len(objects)}\n({counts_str})"
    ax.set_title(title, fontsize=12, pad=10)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved annotated image to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BCCD microscopic images with ground-truth bounding boxes."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/albert/data/bccd",
        help="Path to root BCCD dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to visualize from",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of image in the chosen split to visualize",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default=None,
        help="Specific image name (e.g., BloodImage_00054.jpeg)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick a random image from the split",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save output image file",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive matplotlib window",
    )

    args = parser.parse_args()

    split_dir = Path(args.data_dir) / args.split
    img_dir = split_dir / "img"
    ann_dir = split_dir / "ann"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    images = sorted(list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    if args.image_name:
        image_path = img_dir / args.image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Specified image not found: {image_path}")
    elif args.random:
        image_path = random.choice(images)
    else:
        idx = max(0, min(args.index, len(images) - 1))
        image_path = images[idx]

    ann_path = ann_dir / f"{image_path.name}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    print(f"Loading image: {image_path}")
    print(f"Loading annotation: {ann_path}")

    save_path = args.save_path
    if not save_path and args.no_show:
        # Default save path when no-show is active
        save_path = str(Path(__file__).parent / "ground_truth_sample.png")

    plot_ground_truth(
        image_path=image_path,
        ann_path=ann_path,
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

