import trimesh
import numpy as np


def scale_to_unit_sphere(mesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
        vertices = mesh.vertices - mesh.bounding_box.centroid
        distances = np.linalg.norm(vertices, axis=1)
        vertices /= np.max(distances)
        if isinstance(mesh, trimesh.PointCloud):
            # if the shape is a point cloud, put in vertices only
            return trimesh.PointCloud(vertices=vertices)
        else:
            return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)