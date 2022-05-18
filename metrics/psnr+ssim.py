# Author: Guanxiong
# Compute PSNR + SSIM of given images and save results
# to metrics/results/
# NOTE: first image in config file must be the ground truth!

import argparse
import csv
import yaml
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


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

    with open(f"metrics/results/{config['csv_filename']}", 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        for img_path in config["img_paths"][1:]:
            im_pred_arr = np.array(Image.open(img_path))
            if len(im_pred_arr.shape) == 3 and len(im_gt_arr.shape) == 2:
                # handle the particular case where the GT image contains one channel
                # but the predicted contains three
                im_pred_arr = im_pred_arr[:, :, 0]
                psnr = peak_signal_noise_ratio(im_gt_arr, im_pred_arr)
            else:
                psnr = peak_signal_noise_ratio(im_gt_arr, im_pred_arr)
            if len(im_pred_arr.shape) == 3:
                ssim = structural_similarity(im_gt_arr, im_pred_arr, multichannel=True)
            else:
                ssim = structural_similarity(im_gt_arr, im_pred_arr, multichannel=False)
            resultwriter.writerow([img_path, f"{psnr}", f"{ssim}"])
    print(f"PSNR and SSIM written to metrics/results/{config['csv_filename']}")
