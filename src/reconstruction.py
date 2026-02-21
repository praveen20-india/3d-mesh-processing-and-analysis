"""ICP-based 3D mesh registration and reconstruction pipeline."""

import numpy as np
import open3d as o3d

from .mesh_loader import load_mesh, sample_points
from .sampling_analysis import preprocess_point_cloud, remove_outliers, compute_scale_factor


def prepare_dataset(source_path: str,
                    target_path: str,
                    n_points: int = 20000,
                    voxel_size: float = 0.05) -> dict:
    """Load meshes, sample, clean, scale, and preprocess for registration.

    The pipeline performs:
      1. Mesh loading and uniform point sampling
      2. Centering both clouds at the origin
      3. Statistical outlier removal
      4. Bounding-box scale correction
      5. Voxel downsampling and FPFH feature computation

    Args:
        source_path: Path to the source (reconstructed) STL mesh.
        target_path: Path to the target (ground truth) STL mesh.
        n_points: Number of surface sample points.
        voxel_size: Initial voxel size hint (adjusted dynamically).

    Returns:
        Dictionary with keys: source, target, source_down, target_down,
        source_fpfh, target_fpfh, voxel_size.
    """
    source_mesh = load_mesh(source_path)
    target_mesh = load_mesh(target_path)

    source = sample_points(source_mesh, n_points)
    target = sample_points(target_mesh, n_points)

    # Center both clouds at the origin
    source.translate(-source.get_center())
    target.translate(-target.get_center())

    # Outlier removal
    source = remove_outliers(source)
    target = remove_outliers(target)

    # Scale correction
    scale_factor = compute_scale_factor(source, target)
    source.scale(scale_factor, center=(0, 0, 0))

    # Dynamic voxel sizing based on target extent
    target_bbox = target.get_axis_aligned_bounding_box()
    target_diameter = np.linalg.norm(
        target_bbox.get_max_bound() - target_bbox.get_min_bound())
    adjusted_voxel = target_diameter / 40.0

    # Apply initial perturbation for robustness testing
    trans_init = np.array([
        [0.862, 0.011, -0.507, 0.0],
        [-0.139, 0.967, -0.215, 0.0],
        [0.487, 0.255,  0.835, 0.0],
        [0.0,   0.0,    0.0,   1.0],
    ])
    source.transform(trans_init)

    # Feature extraction
    source_down, source_fpfh = preprocess_point_cloud(source, adjusted_voxel)
    target_down, target_fpfh = preprocess_point_cloud(target, adjusted_voxel)

    return {
        "source": source,
        "target": target,
        "source_down": source_down,
        "target_down": target_down,
        "source_fpfh": source_fpfh,
        "target_fpfh": target_fpfh,
        "voxel_size": adjusted_voxel,
    }


def execute_global_registration(source_down, target_down,
                                source_fpfh, target_fpfh,
                                voxel_size):
    """RANSAC-based global registration using FPFH features.

    Args:
        source_down: Downsampled source point cloud.
        target_down: Downsampled target point cloud.
        source_fpfh: Source FPFH features.
        target_fpfh: Target FPFH features.
        voxel_size: Voxel size for distance threshold computation.

    Returns:
        Open3D RegistrationResult.
    """
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999),
    )
    return result


def execute_fast_global_registration(source_down, target_down,
                                     source_fpfh, target_fpfh,
                                     voxel_size):
    """Fast Global Registration (FGR) using FPFH features.

    Args:
        source_down: Downsampled source point cloud.
        target_down: Downsampled target point cloud.
        source_fpfh: Source FPFH features.
        target_fpfh: Target FPFH features.
        voxel_size: Voxel size for distance threshold computation.

    Returns:
        Open3D RegistrationResult.
    """
    distance_threshold = voxel_size * 0.5

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold),
    )
    return result


def refine_registration(source, target, initial_transformation, voxel_size):
    """Point-to-plane ICP refinement on full-resolution clouds.

    Args:
        source: Full-resolution source point cloud.
        target: Full-resolution target point cloud.
        initial_transformation: Initial alignment (e.g., from RANSAC).
        voxel_size: Voxel size for distance threshold computation.

    Returns:
        Open3D RegistrationResult with refined transformation.
    """
    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return result
