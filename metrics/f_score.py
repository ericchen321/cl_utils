# Author: Chunjin, Guanxiong
# Note: for the threshold, we're using values used in other papers,
# but we might want to pick a different one

import argparse
import yaml
import os, typing
import open3d as o3d
import numpy as np


def calculate_fscore(gt: o3d.geometry.PointCloud, pr: o3d.geometry.PointCloud, th: float = 0.01) -> typing.Tuple[
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
    out_dir = 'metrics/results/'

    cube_side_length = config["cube_side_length"]
    threshold_list = []
    for cube_side_factor in config["cube_side_factors"]:
        threshold_list.append(cube_side_length / cube_side_factor)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    os.makedirs(out_dir, exist_ok=True)
    f_f = open(os.path.join(out_dir, f"fscore_fscore_{config['experiment_name']}.txt"), "w")
    f_p = open(os.path.join(out_dir, f"fscore_precision_{config['experiment_name']}.txt"), "w")
    f_r = open(os.path.join(out_dir, f"fscore_recall_{config['experiment_name']}.txt"), "w")
    
    for meshpath_pred in config["mesh_paths"][1:]:
        gt = np.asarray(o3d.io.read_triangle_mesh(meshpath_gt).vertices)
        pr = np.asarray(o3d.io.read_triangle_mesh(meshpath_pred).vertices)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt)

        pcd_pr = o3d.geometry.PointCloud()
        pcd_pr.points = o3d.utility.Vector3dVector(pr)

        f_f.write(" " + str(meshpath_pred) + "\n")
        f_p.write(" " + str(meshpath_pred) + "\n")
        f_r.write(" " + str(meshpath_pred) + "\n")

        for th in threshold_list:
            f, p, r = calculate_fscore(pcd_gt, pcd_pr, th=th)
            print(meshpath_pred, f'fscore: {str(f)}, precision: {str(p)}, recall: {str(r)}')

            f_f.write(" " + str(f)  + "\n")
            f_p.write(" " + str(p) + "\n")
            f_r.write(" " + str(r) + "\n")

        f_f.write("\n")
        f_p.write("\n")
        f_r.write("\n")

    f_f.close()
    f_p.close()
    f_r.close()
