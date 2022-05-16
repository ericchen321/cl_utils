# Authors: Zhijie, Guanxiong
# Generate normalized predicted/GT meshes;
# Compute Chamfer distance

import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors
from utils.utils import scale_to_unit_sphere
import argparse


def sample_points_from_shape(mesh, num_pts):
    samples = mesh.sample(num_pts) # Sample points (by interpolation) for computation
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
        '--num_pts', type=int, default=1000000, help='number of points to sample for computing CD'
    )
    parser.add_argument(
        '--square_dist', action='store_true', default=False, help='square distance or not'
    )
    args = parser.parse_args()

    pred_mesh, pred_samples, gt_mesh, gt_samples = None, None, None, None
    if args.pred_shape_input != '':
        # Load the mesh
        pred_mesh = trimesh.load(args.pred_shape_input)
        # Normalize
        pred_mesh = scale_to_unit_sphere(pred_mesh)
        # Output normalized shape
        if args.pred_shape_norm_output !='':
            pred_mesh.export(args.pred_shape_norm_output)
        # Sample
        pred_samples, pred_mesh = sample_points_from_shape(pred_mesh, args.num_pts)
    if args.gt_shape_input != '':
        # Load the mesh
        gt_mesh = trimesh.load(args.gt_shape_input)
        # Normalize
        gt_mesh = scale_to_unit_sphere(gt_mesh)
        if args.gt_shape_norm_output != '':
            gt_mesh.export(args.gt_shape_norm_output)
        # Sample
        gt_samples, gt_mesh = sample_points_from_shape(gt_mesh, args.num_pts)

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
