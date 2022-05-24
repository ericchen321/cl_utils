# Authors: Zhijie, Guanxiong
# Generate normalized predicted/GT meshes;
# Compute Chamfer distance

import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors
from utils.utils import scale_to_unit_sphere
import csv
import yaml
import argparse


def sample_points_from_shape(mesh, num_pts):
    samples = mesh.sample(num_pts) # Sample points (by interpolation) for computation
    return samples, mesh


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, help='path to yaml config file', required=True)
    args = p.parse_args()

    config = None
    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Load the GT mesh
    gt_mesh = trimesh.load(config["mesh_paths"][0])
    # Normalize
    gt_mesh = scale_to_unit_sphere(gt_mesh)
    # Sample
    gt_samples, gt_mesh = sample_points_from_shape(gt_mesh, config["num_points_to_sample"])

    cdists = []
    for meshpath_pred in config["mesh_paths"][1:]:
        # Load each predicted mesh
        pred_mesh = trimesh.load(meshpath_pred)
        # Normalize
        if config["normalize"]:
            pred_mesh = scale_to_unit_sphere(pred_mesh)
        # Sample
        pred_samples, pred_mesh = sample_points_from_shape(pred_mesh, config["num_points_to_sample"])

        # Compute Chamfer
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=config["metric"]).fit(pred_samples)
        min_y_to_x = x_nn.kneighbors(gt_samples)[0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=config["metric"]).fit(gt_samples)
        min_x_to_y = y_nn.kneighbors(pred_samples)[0]
        cdist = -1
        if config["square_distances"]:
            cdist = np.mean(np.square(min_y_to_x)) + np.mean(np.square(min_x_to_y))
        else:
            cdist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
        cdists.append(cdist)

    result_filepath = f"metrics/results/chamfer_distance_{config['experiment_name']}.csv"
    with open(result_filepath, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        for meshpath_pred, cdist in zip(config["mesh_paths"][1:], cdists):
            resultwriter.writerow([meshpath_pred, cdist])
    print(f"Chamfer distances written to {result_filepath}")