# Author: Guanxiong
# Compute PSNR of given images and save results
# to metrics/results/
# NOTE: first image in config file must be the ground truth!

import argparse
import csv
import yaml
from PIL import Image
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio


def compute_psnrs(config):
    img_paths_gt = config["gt_img_paths"]
    img_paths_pred = config["pred_img_paths"]
    grayscales = config["grayscales"]

    psnrs = []
    for img_path_gt, img_path_pred, is_gs in zip(
        img_paths_gt, img_paths_pred, grayscales):
        # for each gt-pred pair, compute psnr
        im_gt_arr = np.array(Image.open(img_path_gt))
        im_pred_arr = np.array(Image.open(img_path_pred))
        psnr = None
        if is_gs:
            if len(im_gt_arr.shape) == 3:
                im_gt_arr = im_gt_arr[:, :, 0]
            if len(im_pred_arr.shape) == 3:
                im_pred_arr = im_pred_arr[:, :, 0]
            psnr = peak_signal_noise_ratio(im_gt_arr, im_pred_arr)
        else:
            im_gt_arr = im_gt_arr[:, :, :3]
            im_pred_arr = im_pred_arr[:, :, :3]
            psnr = peak_signal_noise_ratio(im_gt_arr, im_pred_arr)
        psnrs.append(psnr)
    
    return psnrs


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

    result_filename = f"metrics/results/psnr_{config['experiment_name']}.csv"
    with open(result_filename, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        for imgpath_pred, psnr in zip(
            config["pred_img_paths"], psnrs):
            resultwriter.writerow([imgpath_pred, psnr])
    print(f"PSNR written to {result_filename}")