# Author: Chunjin, Guanxiong
# NOTE: larger IoU leads to more accurate result, but is slower
# to compute

import argparse
import csv
import yaml
import numpy as np
import open3d as o3d
from joblib import Parallel, delayed


def iou_pymesh(mesh_src, mesh_pred, scale):
    mesh1 = o3d.io.read_triangle_mesh(mesh_src)
    grid1 = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh1, 0.1/scale, [-1,-1,-1], [1,1,1])
    vertices_grid1 = np.asarray([x.grid_index for x in grid1.get_voxels()])

    mesh2 = o3d.io.read_triangle_mesh(mesh_pred)
    grid2 = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh2, 0.1/scale,[-1,-1,-1], [1,1,1])
    vertices_grid2 = np.asarray([x.grid_index for x in grid2.get_voxels()])

    dim = 20 * scale
    v1 = np.zeros([dim, dim, dim])
    v1[vertices_grid1[:, 0], vertices_grid1[:, 1], vertices_grid1[:, 2]] = 1

    v2 = np.zeros([dim, dim, dim])
    v2[vertices_grid2[:, 0], vertices_grid2[:, 1], vertices_grid2[:, 2]] = 1

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
    out_dir = 'metrics/results/'
    meshpaths_gt = len(config["mesh_paths"][1:]) * [meshpath_gt]
    meshpaths_pred = config["mesh_paths"][1:]
    scale = int(config["scale"])
    scales = len(config["mesh_paths"][1:]) * [scale]
    with Parallel(n_jobs=20) as parallel:
        results = parallel(delayed(iou_pymesh)
            (src_path, pred_path, scale)
            for src_path, pred_path, scale in
            zip(meshpaths_gt, meshpaths_pred, scales))

    result_filepath = f"metrics/results/iou_{config['experiment_name']}.csv"
    with open(result_filepath, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        for meshpth, result in zip(config["mesh_paths"][1:], results):
            resultwriter.writerow([meshpth, result])
    print(f"IoU written to {result_filepath}")