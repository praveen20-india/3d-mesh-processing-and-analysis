"""End-to-end 3D mesh processing and ICP registration pipeline.

This script loads reconstructed meshes from partial image datasets, aligns
them to a ground-truth mesh via global (RANSAC) and local (ICP) registration,
and outputs quantitative fitness/RMSE metrics along with visualizations.

Usage:
    python -m src.main              # Run from project root
    python -m src.main --save-plots # Save result images
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .mesh_loader import load_mesh, sample_points, get_mesh_info
from .sampling_analysis import preprocess_point_cloud, remove_outliers, compute_scale_factor
from .reconstruction import (
    prepare_dataset,
    execute_global_registration,
    refine_registration,
)


# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPORT_IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "result_images")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DATASETS = [
    ("Full",    "full_dataset_mesh.stl"),
    ("Half",    "half_dataset_mesh.stl"),
    ("Quarter", "quarter_dataset_mesh.stl"),
    ("Third",   "third_dataset_mesh.stl"),
]


def render_single_mesh(points, title, save_path, color="steelblue"):
    """Render a single point cloud with matplotlib and save to disk."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=0.4, alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


def render_registration_overlay(source_pts, target_pts, transformation,
                                title, save_path):
    """Render source (orange) overlaid on target (blue) after alignment."""
    R = transformation[:3, :3]
    t = transformation[:3, 3]
    aligned = (R @ source_pts.T).T + t

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(aligned[:, 0], aligned[:, 1], aligned[:, 2],
               c="orange", s=0.4, label="Source (aligned)", alpha=0.5)
    ax.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
               c="steelblue", s=0.4, label="Target (ground truth)", alpha=0.5)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


def render_metric_analysis(labels, fitness_vals, rmse_vals, save_path):
    """Bar chart comparing fitness and RMSE across datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(labels))
    bars1 = ax1.bar(x, fitness_vals, color=["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"],
                    edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Fitness Score", fontsize=12)
    ax1.set_title("ICP Fitness by Dataset", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    for bar, v in zip(bars1, fitness_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{v:.4f}", ha="center", fontsize=10)

    bars2 = ax2.bar(x, rmse_vals, color=["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"],
                    edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Inlier RMSE", fontsize=12)
    ax2.set_title("ICP RMSE by Dataset", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    for bar, v in zip(bars2, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                 f"{v:.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


def render_sampling_comparison(dataset_infos, save_path):
    """Compare mesh sizes (vertex count and diameter) across datasets."""
    labels = [d["label"] for d in dataset_infos]
    vertices = [d["n_vertices"] for d in dataset_infos]
    diameters = [d["diameter"] for d in dataset_infos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(labels))
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

    ax1.bar(x, vertices, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Vertex Count", fontsize=12)
    ax1.set_title("Mesh Vertex Count by Dataset", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    for i, v in enumerate(vertices):
        ax1.text(i, v + max(vertices)*0.02, f"{v:,}", ha="center", fontsize=9)

    ax2.bar(x, diameters, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Bounding Box Diameter", fontsize=12)
    ax2.set_title("Mesh Diameter by Dataset", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    for i, v in enumerate(diameters):
        ax2.text(i, v + max(diameters)*0.02, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3D Mesh Processing and ICP Registration Pipeline")
    parser.add_argument("--save-plots", action="store_true", default=True,
                        help="Save result images to report/result_images/")
    args = parser.parse_args()

    os.makedirs(REPORT_IMG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Use the full-dataset mesh as ground truth reference
    gt_path = os.path.join(DATA_DIR, "full_dataset_mesh.stl")

    # ── 1. Mesh info & sampling comparison ──
    print("\n=== Mesh Dataset Analysis ===\n")
    dataset_infos = []
    for label, fname in DATASETS:
        mesh_path = os.path.join(DATA_DIR, fname)
        mesh = load_mesh(mesh_path)
        info = get_mesh_info(mesh)
        info["label"] = label
        dataset_infos.append(info)
        print(f"  {label:10s}  vertices={info['n_vertices']:>8,}  "
              f"faces={info['n_faces']:>8,}  diameter={info['diameter']:.4f}")

    if args.save_plots:
        render_sampling_comparison(
            dataset_infos,
            os.path.join(REPORT_IMG_DIR, "sampling_comparison.png"))

    # ── 2. Visualize full mesh ──
    print("\n=== Full Mesh Visualization ===\n")
    full_mesh = load_mesh(os.path.join(DATA_DIR, "full_dataset_mesh.stl"))
    full_pcd = sample_points(full_mesh, 30000)
    if args.save_plots:
        render_single_mesh(
            np.asarray(full_pcd.points),
            "Full Dataset -- 3D Surface Point Cloud",
            os.path.join(REPORT_IMG_DIR, "full_mesh.png"))

    # ── 3. Visualize partial mesh ──
    print("\n=== Partial Mesh Comparison ===\n")
    partial_names = [("Half", "half_dataset_mesh.stl"),
                     ("Quarter", "quarter_dataset_mesh.stl"),
                     ("Third", "third_dataset_mesh.stl")]
    if args.save_plots:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                                 subplot_kw={"projection": "3d"})
        partial_colors = ["#f39c12", "#e74c3c", "#9b59b6"]
        for ax, (label, fname), color in zip(axes, partial_names, partial_colors):
            mesh = load_mesh(os.path.join(DATA_DIR, fname))
            pts = np.asarray(sample_points(mesh, 20000).points)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=color, s=0.3, alpha=0.5)
            ax.set_title(f"{label} Dataset", fontsize=12, fontweight="bold")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.grid(True, alpha=0.3)
        plt.suptitle("Partial Dataset Surface Comparisons", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_IMG_DIR, "partial_mesh.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [saved] partial_mesh.png")

    # ── 4. ICP registration on all datasets ──
    print("\n=== ICP Registration Pipeline ===\n")
    print(f"{'Dataset':<10} | {'Fitness':<10} | {'RMSE':<10}")
    print("-" * 36)

    labels_list = []
    fitness_list = []
    rmse_list = []
    results_text = []
    results_text.append("3D Mesh Processing -- ICP Registration Results")
    results_text.append("=" * 50)
    results_text.append(f"{'Dataset':<10} {'Fitness':<12} {'RMSE':<12}")
    results_text.append("-" * 40)

    for label, fname in DATASETS:
        source_path = os.path.join(DATA_DIR, fname)

        data = prepare_dataset(source_path, gt_path, n_points=20000)

        result_ransac = execute_global_registration(
            data["source_down"], data["target_down"],
            data["source_fpfh"], data["target_fpfh"],
            data["voxel_size"])

        result_icp = refine_registration(
            data["source"], data["target"],
            result_ransac.transformation, data["voxel_size"])

        fitness = result_icp.fitness
        rmse = result_icp.inlier_rmse

        print(f"{label:<10} | {fitness:<10.4f} | {rmse:<10.4f}")
        labels_list.append(label)
        fitness_list.append(fitness)
        rmse_list.append(rmse)
        results_text.append(f"{label:<10} {fitness:<12.6f} {rmse:<12.6f}")

        # Save reconstruction overlay for each dataset
        if args.save_plots:
            save_name = f"reconstruction_{label.lower()}.png" if label != "Full" else "reconstruction_result.png"
            render_registration_overlay(
                np.asarray(data["source"].points),
                np.asarray(data["target"].points),
                result_icp.transformation,
                f"{label} Dataset -- ICP Registration Result",
                os.path.join(REPORT_IMG_DIR, save_name))

    # ── 5. Metric analysis chart ──
    if args.save_plots:
        render_metric_analysis(
            labels_list, fitness_list, rmse_list,
            os.path.join(REPORT_IMG_DIR, "metric_analysis.png"))

    # ── 6. Write numerical results ──
    results_text.append("-" * 40)
    results_text.append("")
    results_text.append("Dataset mesh statistics:")
    for info in dataset_infos:
        results_text.append(
            f"  {info['label']:<10} vertices={info['n_vertices']:>8,}  "
            f"faces={info['n_faces']:>8,}  diameter={info['diameter']:.4f}")

    results_path = os.path.join(RESULTS_DIR, "numerical_results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(results_text) + "\n")
    print(f"\n[saved] {results_path}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
