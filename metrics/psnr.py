# Author: Guanxiong
# Compute PSNR of given images and save results
# to metrics/results/

import argparse
import csv
import yaml
from utils.utils import get_image_paths_and_specs, compute_metric_2d


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

    img_paths_gt, img_paths_pred_dict, grayscales = get_image_paths_and_specs(config)
    metric_name = "psnr"
    metrics_dict = compute_metric_2d(img_paths_gt, img_paths_pred_dict, grayscales, metric_name)

    result_filename = f"metrics/results/{metric_name}_{config['experiment_name']}.csv"
    with open(result_filename, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        resultwriter.writerow(["baseline", "predicted image path", metric_name])
        for baseline, metrics in metrics_dict.items():
            for img_path_pred, psnr in zip(
                img_paths_pred_dict[baseline], metrics):
                resultwriter.writerow([baseline, img_path_pred, psnr])
    print(f"{metric_name} written to {result_filename}")