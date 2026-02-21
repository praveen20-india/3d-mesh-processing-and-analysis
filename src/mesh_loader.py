"""Mesh loading and basic geometry utilities."""

import os
import numpy as np
import open3d as o3d


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    """Load a triangle mesh from an STL file.

    Args:
        path: Path to the STL file.

    Returns:
        Open3D TriangleMesh object.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the mesh contains no triangles.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = o3d.io.read_triangle_mesh(path)

    if not mesh.has_triangles():
        raise RuntimeError(f"Mesh contains no triangles: {path}")

    return mesh


def sample_points(mesh: o3d.geometry.TriangleMesh,
                  n_points: int = 20000) -> o3d.geometry.PointCloud:
    """Uniformly sample points from a mesh surface.

    Args:
        mesh: Input triangle mesh.
        n_points: Number of points to sample.

    Returns:
        Point cloud with uniformly distributed samples.
    """
    return mesh.sample_points_uniformly(number_of_points=n_points)


def get_mesh_info(mesh: o3d.geometry.TriangleMesh) -> dict:
    """Extract basic geometric properties of a mesh.

    Args:
        mesh: Input triangle mesh.

    Returns:
        Dictionary with vertex count, face count, bounding box extents,
        and maximum diameter.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    bbox = mesh.get_axis_aligned_bounding_box()
    extents = bbox.get_max_bound() - bbox.get_min_bound()
    diameter = np.linalg.norm(extents)

    return {
        "n_vertices": len(vertices),
        "n_faces": len(triangles),
        "bbox_extents": extents,
        "diameter": diameter,
    }
