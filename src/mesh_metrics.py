"""Quantitative metrics for mesh registration and reconstruction evaluation."""

import numpy as np


def compute_fitness_summary(result) -> dict:
    """Extract fitness and RMSE from an Open3D registration result.

    Args:
        result: Open3D RegistrationResult object.

    Returns:
        Dictionary with fitness score, inlier RMSE, and transformation matrix.
    """
    return {
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
        "transformation": result.transformation,
    }


def compute_position_errors(gt_positions: dict,
                            sfm_positions: dict,
                            ref_label_a: str,
                            ref_label_b: str) -> dict:
    """Compute normalized relative position errors.

    Positions are normalized by the reference distance between two
    designated cameras (ref_label_a and ref_label_b) in each coordinate
    system. The error is the relative difference of the normalized
    distances to ref_label_a.

    Args:
        gt_positions: Mapping from label to 3D positions (ground truth).
        sfm_positions: Mapping from label to 3D positions (estimated).
        ref_label_a: First reference camera label.
        ref_label_b: Second reference camera label.

    Returns:
        Dictionary mapping each label to its relative position error in percent.
    """
    t_a_gt = gt_positions[ref_label_a]
    t_b_gt = gt_positions[ref_label_b]
    t_a_sfm = sfm_positions[ref_label_a]
    t_b_sfm = sfm_positions[ref_label_b]

    d_gt = np.linalg.norm(t_b_gt - t_a_gt)
    d_sfm = np.linalg.norm(t_b_sfm - t_a_sfm)

    errors = {}
    for label in sfm_positions:
        if label not in gt_positions:
            continue

        d_i_gt = np.linalg.norm(gt_positions[label] - t_a_gt) / d_gt
        d_i_sfm = np.linalg.norm(sfm_positions[label] - t_a_sfm) / d_sfm

        if d_i_gt < 1e-8:
            errors[label] = 0.0
        else:
            errors[label] = abs(d_i_sfm - d_i_gt) / d_i_gt * 100.0

    return errors


def compute_orientation_errors(gt_rotations: dict,
                               sfm_rotations: dict,
                               alignment_label: str) -> dict:
    """Compute orientation errors after global alignment.

    A single alignment rotation is computed from the designated camera
    (alignment_label) and applied to all SfM rotations before measuring
    the angular difference to ground truth.

    Args:
        gt_rotations: Mapping from label to 3x3 rotation matrices (ground truth).
        sfm_rotations: Mapping from label to 3x3 rotation matrices (estimated).
        alignment_label: Camera label used to compute the global alignment.

    Returns:
        Dictionary mapping each label to orientation error in degrees.
    """
    from scipy.spatial.transform import Rotation as R

    q_sfm_ref = R.from_matrix(sfm_rotations[alignment_label])
    q_gt_ref = R.from_matrix(gt_rotations[alignment_label])
    q_alignment = q_gt_ref * q_sfm_ref.inv()

    errors = {}
    for label in sfm_rotations:
        if label not in gt_rotations:
            continue

        q_sfm = R.from_matrix(sfm_rotations[label])
        q_gt = R.from_matrix(gt_rotations[label])
        q_sfm_aligned = q_alignment * q_sfm
        q_err = q_gt * q_sfm_aligned.inv()

        errors[label] = q_err.magnitude() * (180.0 / np.pi)

    return errors
