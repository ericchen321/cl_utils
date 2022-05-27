# Author: Chunjin, Guanxiong
# Note: for the threshold, we're using values used in other papers,
# but we might want to pick a different one

import argparse
import yaml
import os, typing
import open3d as o3d
import numpy as np
import csv
from utils.utils import scale_to_unit_sphere


def calculate_fscore(
    gt: o3d.geometry.PointCloud, pr: o3d.geometry.PointCloud, th: float = 0.01) -> typing.Tuple[
    float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall


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

    threshold = config["cube_side_length"] / config["cube_side_factor"]

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    out_dir = 'metrics/results/'
    os.makedirs(out_dir, exist_ok=True)
    result_filepath = f"{out_dir}/f_score_{config['experiment_name']}.csv"
    with open(result_filepath, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')

        for meshpath_pred in config["mesh_paths"][1:]:
            mesh_gt = o3d.io.read_triangle_mesh(meshpath_gt)
            mesh_pred = o3d.io.read_triangle_mesh(meshpath_pred)

            # normalize if needed
            if config["normalize"]:
                mesh_gt = scale_to_unit_sphere(mesh_gt)
                mesh_pred = scale_to_unit_sphere(mesh_pred)

            gt = np.asarray(mesh_gt.vertices)
            pr = np.asarray(mesh_pred.vertices)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt)

            pcd_pr = o3d.geometry.PointCloud()
            pcd_pr.points = o3d.utility.Vector3dVector(pr)

            f_score, precision, recall = calculate_fscore(pcd_gt, pcd_pr, th=threshold)
            resultwriter.writerow([meshpath_pred, f_score, precision, recall])
    print(f"F Score, Precision, Recall written to {result_filepath}")
