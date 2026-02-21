"""Point cloud preprocessing and sampling analysis utilities."""

import numpy as np
import open3d as o3d


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud,
                           voxel_size: float) -> tuple:
    """Downsample, estimate normals, and compute FPFH features.

    Args:
        pcd: Input point cloud.
        voxel_size: Voxel size for downsampling.

    Returns:
        Tuple of (downsampled point cloud, FPFH feature descriptor).
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


def remove_outliers(pcd: o3d.geometry.PointCloud,
                    nb_neighbors: int = 20,
                    std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """Remove statistical outliers from a point cloud.

    Points whose mean distance to neighbors exceeds ``std_ratio`` standard
    deviations from the global mean are removed.

    Args:
        pcd: Input point cloud.
        nb_neighbors: Number of nearest neighbors for statistics.
        std_ratio: Standard deviation multiplier threshold.

    Returns:
        Cleaned point cloud.
    """
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)


def compute_scale_factor(source: o3d.geometry.PointCloud,
                         target: o3d.geometry.PointCloud) -> float:
    """Compute the bounding-box diameter ratio between target and source.

    This scale factor can be applied to the source cloud to match
    the spatial extent of the target.

    Args:
        source: Source point cloud.
        target: Target point cloud.

    Returns:
        Scale factor (target_diameter / source_diameter).
    """
    source_bbox = source.get_axis_aligned_bounding_box()
    target_bbox = target.get_axis_aligned_bounding_box()

    source_diameter = np.linalg.norm(
        source_bbox.get_max_bound() - source_bbox.get_min_bound())
    target_diameter = np.linalg.norm(
        target_bbox.get_max_bound() - target_bbox.get_min_bound())

    return target_diameter / source_diameter
