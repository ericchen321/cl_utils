# Authors: Zhijie, Guanxiong
# Generate normalized predicted/GT meshes;
# Compute Chamfer distance

import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors
from utils import scale_to_unit_sphere
import argparse


def sample_points_from_shape(shape_in_path, num_pts):
    mesh = trimesh.load(shape_in_path)
    mesh = scale_to_unit_sphere(mesh)

    num_verts = mesh.vertices.shape[0]
    num_pts = min(num_pts, num_verts)
    print(f"sampling {num_pts} points from shape {shape_in_path}")
    if isinstance(mesh, trimesh.PointCloud):
        # if pc, sample manually
        sampled_ids = np.random.choice(num_verts, size=num_pts, replace=False)
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
    parser.add_argument(
        '--square_dist', action='store_true', default=False, help='square distance or not'
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
        if args.square_dist:
            cdist = np.mean(np.square(min_y_to_x)) + np.mean(np.square(min_x_to_y))
        else:
            cdist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
        print("Chamfer distance: {}".format(cdist))
