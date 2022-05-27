import trimesh
import open3d as o3d
import numpy as np


def scale_to_unit_sphere(mesh):
    r"""
    Normalize a mesh to the unit sphere.

    :param mesh: Trimesh/Open3d mesh or point cloud
    
    Return:
        scaled mesh in its original format
    """

    if isinstance(mesh, trimesh.Scene) \
        or isinstance(mesh, trimesh.Trimesh) \
        or isinstance(mesh, trimesh.PointCloud):
        # deal with shape in Trimesh format
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()

        vertices = mesh.vertices - mesh.bounding_box.centroid
        distances = np.linalg.norm(vertices, axis=1)
        vertices /= np.max(distances)

        if isinstance(mesh, trimesh.Trimesh):
            return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
        elif isinstance(mesh, trimesh.PointCloud):
            return trimesh.PointCloud(vertices=vertices)
        else:
            raise NotImplementedError
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        # deal with shape shape in Open3d format
        centroid = mesh.get_axis_aligned_bounding_box().get_center()
        vertices = np.asarray(mesh.vertices) - centroid
        distances = np.linalg.norm(vertices, axis=1)
        vertices /= np.max(distances)
        return o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=mesh.triangles
        )
    else:
        raise NotImplementedError
