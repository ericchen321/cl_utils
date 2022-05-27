# Author: Chunjin, Guanxiong
# NOTE: larger IoU leads to more accurate result, but is slower
# to compute

import argparse
import csv
import yaml
import numpy as np
import open3d as o3d
from joblib import Parallel, delayed
from utils.utils import scale_to_unit_sphere
import os


def iou_pymesh(meshpath_gt, meshpath_pred, scale, normalize):
    mesh_gt = o3d.io.read_triangle_mesh(meshpath_gt)
    if normalize:
        mesh_gt = scale_to_unit_sphere(mesh_gt)
    grid_gt = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh_gt, 0.1/scale, [-1,-1,-1], [1,1,1])
    vertices_grid_gt = np.asarray([x.grid_index for x in grid_gt.get_voxels()])

    mesh_pred = o3d.io.read_triangle_mesh(meshpath_pred)
    if normalize:
        mesh_pred = scale_to_unit_sphere(mesh_pred)
    grid_pred = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh_pred, 0.1/scale,[-1,-1,-1], [1,1,1])
    vertices_grid_pred = np.asarray([x.grid_index for x in grid_pred.get_voxels()])

    dim = 20 * scale
    v1 = np.zeros([dim, dim, dim])
    v1[vertices_grid_gt[:, 0], vertices_grid_gt[:, 1], vertices_grid_gt[:, 2]] = 1

    v2 = np.zeros([dim, dim, dim])
    v2[vertices_grid_pred[:, 0], vertices_grid_pred[:, 1], vertices_grid_pred[:, 2]] = 1

    intersection = np.sum(np.logical_and(v1, v2))
    union = np.sum(np.logical_or(v1, v2))
    return float(intersection) / union


if __name__=='__main__':  
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, help='path to yaml config file', required=True)
    args = p.parse_args()

    config = None
    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    meshpath_gt = config["mesh_paths"][0]
    meshpaths_gt = len(config["mesh_paths"][1:]) * [meshpath_gt]
    meshpaths_pred = config["mesh_paths"][1:]
    scale = int(config["scale"])
    scales = len(config["mesh_paths"][1:]) * [scale]
    normalize_list = len(config["mesh_paths"][1:]) * [config["normalize"]]
    with Parallel(n_jobs=20) as parallel:
        results = parallel(delayed(iou_pymesh)
            (meshpath_gt, meshpath_pred, scale, normalize)
            for meshpath_gt, meshpath_pred, scale, normalize in
            zip(meshpaths_gt, meshpaths_pred, scales, normalize_list))

    out_dir = 'metrics/results/'
    os.makedirs(out_dir, exist_ok=True)
    result_filepath = f"{out_dir}/iou_{config['experiment_name']}.csv"
    with open(result_filepath, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        for meshpth, result in zip(config["mesh_paths"][1:], results):
            resultwriter.writerow([meshpth, result])
    print(f"IoU written to {result_filepath}")