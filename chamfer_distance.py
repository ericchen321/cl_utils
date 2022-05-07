# Authors: Zhijie, Guanxiong
# Generate normalized predicted/GT meshes;
# Compute Chamfer distance

import numpy as np
import trimesh
import sys
# import torch
from sklearn.neighbors import NearestNeighbors
import argparse


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

def sample_points_from_shape(shape_in_path, num_pts):
    mesh = trimesh.load(shape_in_path)
    mesh = scale_to_unit_sphere(mesh)

    if isinstance(mesh, trimesh.PointCloud):
        # if pc, sample manually
        num_verts = mesh.vertices.shape[0]
        sampled_ids = np.random.randint(0, num_verts, num_pts)
        samples = mesh.vertices[sampled_ids, :]
    else:
        samples = mesh.sample(num_pts) # Sample a subset of points for computation
    return samples, mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred_shape_input', type=str, default='')
    parser.add_argument(
        '--gt_shape_input', type=str, default='')
    parser.add_argument(
        '--pred_shape_norm_output', type=str, default='')
    parser.add_argument(
        '--gt_shape_norm_output', type=str, default='')
    parser.add_argument(
        '--metric', type=str, default='l2')
    parser.add_argument(
        '--num_pts', type=int, default=1000000
    )
    args = parser.parse_args()

    if args.pred_shape_input != '':
        pred_samples, pred_mesh = sample_points_from_shape(args.pred_shape_input, args.num_pts)
    if args.gt_shape_input != '':
        gt_samples, gt_mesh = sample_points_from_shape(args.gt_shape_input, args.num_pts)

    # Output normalized gt && output shapes
    if args.pred_shape_norm_output !='':
        pred_mesh.export(args.pred_shape_norm_output)
    if args.gt_shape_norm_output != '':
        gt_mesh.export(args.gt_shape_norm_output)

    # Compute Chamfer
    if (args.pred_shape_input != '' and args.gt_shape_input != ''):
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=args.metric).fit(pred_samples)
        min_y_to_x = x_nn.kneighbors(gt_samples)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=args.metric).fit(gt_samples)
        min_x_to_y = y_nn.kneighbors(pred_samples)[0]
        cdist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
        print("Chamfer distance: {}".format(cdist))
