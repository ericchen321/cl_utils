# Author: Guanxiong
# Compute 2D metrics and write to csv file
# NOTE: first image in config file must be the ground truth!

import argparse
import csv
import yaml
from psnr import compute_psnrs
from ssim import compute_ssims


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

    psnrs = compute_psnrs(config)
    ssims = compute_ssims(config)
    result_filename = f"metrics/results/2d_{config['experiment_name']}.csv"
    with open(result_filename, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        resultwriter.writerow(["path", "psnr", "ssim"])
        for imgpath_pred, psnr, ssim in zip(
            config["img_paths"][1:], psnrs, ssims):
            resultwriter.writerow([imgpath_pred, psnr, ssim])
    print(f"2D metrics written to {result_filename}")