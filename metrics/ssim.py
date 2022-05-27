# Author: Guanxiong
# Compute SSIM of given images and save results
# to metrics/results/
# NOTE: first image in config file must be the ground truth!

import argparse
import csv
import yaml
from PIL import Image
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity


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

    img_path_gt = config["img_paths"][0]
    im_gt_arr = np.array(Image.open(img_path_gt))
    if config["grayscale"] and len(im_gt_arr.shape) == 3:
        im_gt_arr = im_gt_arr[:, :, 0]

    result_filename = f"metrics/results/ssim_{config['experiment_name']}.csv"
    with open(result_filename, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        for img_path in config["img_paths"][1:]:
            im_pred_arr = np.array(Image.open(img_path))
            
            if config["grayscale"]:
                if len(im_pred_arr.shape) == 3:
                    im_pred_arr = im_pred_arr[:, :, 0]
                ssim = structural_similarity(im_gt_arr, im_pred_arr, multichannel=False)
            else:
                im_gt_arr = im_gt_arr[:, :, :3]
                im_pred_arr = im_pred_arr[:, :, :3]
                ssim = structural_similarity(im_gt_arr, im_pred_arr, multichannel=True)
            
            resultwriter.writerow([img_path, ssim])
    print(f"SSIM written to {result_filename}")