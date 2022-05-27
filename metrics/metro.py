# Author: Xindong, Guanxiong

import numpy as np
import trimesh
import csv
import yaml
import argparse
from scipy.spatial.distance import directed_hausdorff
from utils.utils import scale_to_unit_sphere


def sample_points_from_shape(mesh, num_pts):
    samples = mesh.sample(num_pts) # Sample points (by interpolation) for computation
    return samples

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
    meshpaths_pred = config["mesh_paths"][1:]
    num_samples = config["num_samples"]
    gt_mesh = trimesh.load(meshpath_gt)
    if config["normalize"]:
        gt_mesh = scale_to_unit_sphere(gt_mesh)
    gt_samples = gt_mesh.sample(num_samples)
    
    result_filepath = f"metrics/results/metro_{config['experiment_name']}.csv"
    with open(result_filepath, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        for meshpth in meshpaths_pred:
            pr_mesh = trimesh.load(meshpth)
            if config["normalize"]:
                pr_mesh = scale_to_unit_sphere(pr_mesh)
            pr_samples = pr_mesh.sample(num_samples)
            result = max(directed_hausdorff(gt_samples, pr_samples)[0], directed_hausdorff(pr_samples, gt_samples)[0])
            resultwriter.writerow([meshpth, result])
    print(f"Metro written to {result_filepath}")