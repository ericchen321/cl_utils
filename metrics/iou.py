# Author: Guanxiong
# NOTE: larger IoU leads to more accurate result, but is slower
# to compute

import csv
import yaml
import argparse
from utils.utils import get_mesh_paths
from metrics.sdf import compute_metric_3d


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

    mesh_paths_gt, mesh_paths_pred_dict = get_mesh_paths(config)
    metric_name = "iou"
    metrics_dict = compute_metric_3d(
        mesh_paths_gt, mesh_paths_pred_dict, metric_name, config)

    result_filepath = f"metrics/results/{metric_name}_{config['experiment_name']}.csv"
    with open(result_filepath, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        resultwriter.writerow(["baseline", "predicted mesh path", metric_name])
        for baseline, metrics in metrics_dict.items():
            for mesh_path_pred, cd in zip(
                mesh_paths_pred_dict[baseline], metrics):
                resultwriter.writerow([baseline, mesh_path_pred, cd])
    print(f"{metric_name} written to {result_filepath}")