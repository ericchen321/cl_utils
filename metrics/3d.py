# Author: Guanxiong
# Compute 3D metrics and write to csv file
# NOTE: first image in config file must be the ground truth!

import argparse
import csv
import yaml
from chamfer_distance import compute_chamfers
from f_score import compute_fscores
from iou import compute_ious
from metro import compute_metros


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

    cdists = compute_chamfers(config)
    f_scores, precisions, recalls = compute_fscores(config)
    ious = compute_ious(config)
    metros = compute_metros(config)

    result_filepath = f"metrics/results/3d_{config['experiment_name']}.csv"
    with open(result_filepath, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        resultwriter.writerow(
            ["path", "chamfer distance", "f-score", "precision", "recall", "iou", "metro"])
        for meshpath_pred, cdist, f_score, precision, recall, iou, metro in zip(
            config["mesh_paths"][1:], cdists, f_scores, precisions, recalls, ious, metros):
            resultwriter.writerow([meshpath_pred, cdist, f_score, precision, recall, iou, metro])
    print(f"3D metrics written to {result_filepath}")