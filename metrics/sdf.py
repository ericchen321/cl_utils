# Author: Chunjin, Xindong, Zhijie, Guanxiong
# Compute 3D metrics and write to csv file
# NOTE: first image in config file must be the ground truth!

import argparse
import csv
import yaml
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils.utils import scale_to_unit_sphere
from scipy.spatial.distance import directed_hausdorff
import open3d as o3d
from joblib import Parallel, delayed
from utils.utils import get_mesh_paths


def sample_points_from_shape(mesh, num_pts):
    samples = mesh.sample(num_pts) # Sample points (by interpolation) for computation
    return samples, mesh


def compute_chamfer(mesh_path_gt, mesh_path_pred, config):
    # Load the GT mesh
    mesh_gt = trimesh.load(mesh_path_gt)
    # Normalize
    mesh_gt = scale_to_unit_sphere(mesh_gt)
    # Sample
    gt_samples, mesh_gt = sample_points_from_shape(mesh_gt, config["num_points_to_sample"])

    # Load the predicted mesh
    mesh_pred = trimesh.load(mesh_path_pred)
    # Normalize
    if config["normalize"]:
        mesh_pred = scale_to_unit_sphere(mesh_pred)
    # Sample
    pred_samples, mesh_pred = sample_points_from_shape(mesh_pred, config["num_points_to_sample"])

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

    return cdist


def compute_metro(mesh_path_gt, mesh_path_pred, config):
    # Load the GT mesh
    mesh_gt = trimesh.load(mesh_path_gt)
    # Normalize
    mesh_gt = scale_to_unit_sphere(mesh_gt)
    # Sample
    gt_samples = mesh_gt.sample(config["num_points_to_sample"])

    # Load the predicted mesh
    mesh_pred = trimesh.load(mesh_path_pred)
    # Normalize
    if config["normalize"]:
        mesh_pred = scale_to_unit_sphere(mesh_pred)
    # Sample
    pred_samples = mesh_pred.sample(config["num_points_to_sample"])
    
    # Compute Metro
    metro = max(directed_hausdorff(gt_samples, pred_samples)[0], directed_hausdorff(pred_samples, gt_samples)[0])
    
    return metro


def compute_f_score(mesh_path_gt, mesh_path_pred, config):
    threshold = config["cube_side_length"] / config["cube_side_factor"]
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Load and normalize GT mesh
    mesh_gt = o3d.io.read_triangle_mesh(mesh_path_gt)
    mesh_gt = scale_to_unit_sphere(mesh_gt)

    # Load (and normalize) pred mesh
    mesh_pred = o3d.io.read_triangle_mesh(mesh_path_pred)
    if config["normalize"]:
        mesh_pred = scale_to_unit_sphere(mesh_pred)

    # Extract PCs
    gt = np.asarray(mesh_gt.vertices)
    pr = np.asarray(mesh_pred.vertices)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt)
    pcd_pr = o3d.geometry.PointCloud()
    pcd_pr.points = o3d.utility.Vector3dVector(pr)
    
    d1 = pcd_gt.compute_point_cloud_distance(pcd_pr)
    d2 = pcd_pr.compute_point_cloud_distance(pcd_gt)
    if len(d1) and len(d2):
        recall = float(sum(d < threshold for d in d2)) / float(len(d2))
        precision = float(sum(d < threshold for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0

    return fscore


def iou_pymesh(mesh_path_gt, mesh_path_pred, scale, normalize):
    mesh_gt = o3d.io.read_triangle_mesh(mesh_path_gt)
    if normalize:
        mesh_gt = scale_to_unit_sphere(mesh_gt)
    grid_gt = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh_gt, 0.1/scale, [-1,-1,-1], [1,1,1])
    vertices_grid_gt = np.asarray([x.grid_index for x in grid_gt.get_voxels()])

    mesh_pred = o3d.io.read_triangle_mesh(mesh_path_pred)
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


def compute_metric_3d(mesh_paths_gt, mesh_paths_pred_dict, metric_name, config):
    metrics = {}
        
    for baseline, mesh_paths_pred in mesh_paths_pred_dict.items():
        # initialize metrics as empty lists
        metrics[baseline] = []
        
        # for each baseline, compute metric for all meshes
        if metric_name != "iou":
            for mesh_path_gt, mesh_path_pred in zip(
                mesh_paths_gt, mesh_paths_pred):
                    # for each gt-pred pair, compute the metric
                    metric = None
                    if metric_name == "cd":
                        metric = compute_chamfer(mesh_path_gt, mesh_path_pred, config)
                    elif metric_name == "metro":
                        metric = compute_metro(mesh_path_gt, mesh_path_pred, config)
                    elif metric_name == "f_score":
                        metric = compute_f_score(mesh_path_gt, mesh_path_pred, config)
                    else:
                        raise NotImplementedError
                    metrics[baseline].append(metric)
        else:
            # Parallelize IoU computation
            scales = len(mesh_paths_pred) * [int(config["scale"])]
            normalize_list = len(mesh_paths_pred) * [config["normalize"]]

            with Parallel(n_jobs=20) as parallel:
                ious = parallel(delayed(iou_pymesh)
                    (mesh_path_gt, mesh_path_pred, scale, normalize)
                    for mesh_path_gt, mesh_path_pred, scale, normalize in
                    zip(mesh_paths_gt, mesh_paths_pred, scales, normalize_list))
                metrics[baseline] = ious

    return metrics


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

    # compute each metric individually
    mesh_paths_gt, mesh_paths_pred_dict = get_mesh_paths(config)
    metric_names = ["cd", "metro", "f_score", "iou"]
    metrics_dicts = []
    for metric_name in metric_names:
        per_metric_dict = compute_metric_3d(
            mesh_paths_gt, mesh_paths_pred_dict, metric_name, config)
        metrics_dicts.append(per_metric_dict)

    # pack into a single dict
    metrics_dict = {}
    baselines = list(metrics_dicts[0].keys())
    for baseline in baselines:
        metrics_dict[baseline] = {}
        for i in range(len(metric_names)):
            metrics_dict[baseline][metric_names[i]] = metrics_dicts[i][baseline]

    # write results to file
    result_filepath = f"metrics/results/sdf_{config['experiment_name']}.csv"
    with open(result_filepath, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        resultwriter.writerow(["baseline", "predicted image path"] + metric_names)
        for baseline, metrics in metrics_dict.items():
            # for each baseline, write all metrics
            for mesh_path_pred, cd, metro, f_score, iou in zip(
                mesh_paths_pred_dict[baseline],
                metrics["cd"],
                metrics["metro"],
                metrics["f_score"],
                metrics["iou"]):
                resultwriter.writerow(
                    [baseline, mesh_path_pred, cd, metro, f_score, iou])
    print(f"sdf metrics written to {result_filepath}")