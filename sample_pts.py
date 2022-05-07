import os
import shutil
import h5py
import numpy as np
import trimesh



##### ====================================
# TODO For SIREN - Sample points and normals for a shape
shape_path = "/home/eric/cpsc533r/project/bacon/experiments/outputs/meshes/bacon_armadillo.obj"
out_shape_path = "/home/eric/cpsc533r/project/bacon/experiments/outputs/meshes/bacon_armadillo.xyz"

mesh = trimesh.load(shape_path)

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

# mesh = scale_to_unit_sphere(mesh)
samples, fid  = mesh.sample(3000000, return_index=True)
# compute the barycentric coordinates of each sample
bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[fid], points=samples)
# interpolate vertex normals from barycentric coordinates
interp = trimesh.unitize((mesh.vertex_normals[mesh.faces[fid]] * trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
out_data = np.concatenate([samples, interp], axis=-1)

np.savetxt(out_shape_path, out_data, fmt='%1.6f')

