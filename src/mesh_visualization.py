"""Mesh and point cloud visualization utilities."""

import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def draw_registration_result(source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             transformation: np.ndarray) -> None:
    """Visualize registration result with colored point clouds.

    Source is rendered in orange and target in blue. The transformation
    is applied to the source before display.

    Args:
        source: Source point cloud.
        target: Target point cloud.
        transformation: 4x4 homogeneous transformation matrix.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        window_name="Registration Result",
        width=1024,
        height=768,
        left=50,
        top=50,
    )


def visualize_mesh_matplotlib(points: np.ndarray,
                              title: str = "Point Cloud",
                              save_path: str = None,
                              color: str = "steelblue",
                              point_size: float = 0.5,
                              alpha: float = 0.6) -> None:
    """Render a 3D point cloud using matplotlib.

    Args:
        points: (N, 3) array of 3D coordinates.
        title: Plot title.
        save_path: If provided, save the figure to this path.
        color: Marker color.
        point_size: Marker size.
        alpha: Marker transparency.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=point_size, alpha=alpha)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def visualize_registration_matplotlib(source: o3d.geometry.PointCloud,
                                      target: o3d.geometry.PointCloud,
                                      transformation: np.ndarray,
                                      title: str = "Registration Result",
                                      save_path: str = None) -> None:
    """Render registration overlay using matplotlib.

    Args:
        source: Source point cloud.
        target: Target point cloud.
        transformation: 4x4 transformation applied to source.
        title: Plot title.
        save_path: If provided, save the figure to this path.
    """
    source_pts = np.asarray(source.points)
    target_pts = np.asarray(target.points)

    R = transformation[:3, :3]
    t = transformation[:3, 3]
    source_pts_transformed = (R @ source_pts.T).T + t

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(source_pts_transformed[:, 0],
               source_pts_transformed[:, 1],
               source_pts_transformed[:, 2],
               c="orange", s=0.5, label="Source (aligned)", alpha=0.6)

    ax.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
               c="steelblue", s=0.5, label="Target (ground truth)", alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
