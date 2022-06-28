# Author: Guanxiong
# Compute 2D metrics and write to csv file

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

    # compute each metric individually
    img_paths_gt, img_paths_pred_dict, grayscales = get_image_paths_and_specs(config)
    psnrs_dict = compute_metric_2d(img_paths_gt, img_paths_pred_dict, grayscales, "psnr")
    ssims_dict = compute_metric_2d(img_paths_gt, img_paths_pred_dict, grayscales, "ssim")

    # pack into a single dict
    metrics_dict = {}
    baselines = list(psnrs_dict.keys())
    for baseline in baselines:
        metrics_dict[baseline] = {}
        metrics_dict[baseline]["psnr"] = psnrs_dict[baseline]
        metrics_dict[baseline]["ssim"] = ssims_dict[baseline]

    # write results to file
    result_filename = f"metrics/results/2d_{config['experiment_name']}.csv"
    with open(result_filename, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        resultwriter.writerow(["baseline", "predicted image path", "psnr", "ssim"])
        for baseline, metrics in metrics_dict.items():
            # for each baseline, write all metrics
            for img_path_pred, psnr, ssim in zip(
                img_paths_pred_dict[baseline],
                metrics["psnr"],
                metrics["ssim"]):
                resultwriter.writerow([baseline, img_path_pred, psnr, ssim])
    print(f"2d metrics written to {result_filename}")