# Author: Guanxiong
# Compute 2D metrics and write to csv file

import argparse
import csv
import yaml
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from utils.utils import get_image_paths_and_specs


def compute_metric_2d(img_paths_gt, img_paths_pred_dict, grayscales, metric_name):
    metrics = {}
    
    # set metric function
    metric_fn = None
    if metric_name == "psnr":
        metric_fn = peak_signal_noise_ratio
    elif metric_name == "ssim":
        metric_fn = structural_similarity
    else:
        raise NotImplementedError

    for baseline, img_paths_pred in img_paths_pred_dict.items():
        # initialize metrics as empty lists
        metrics[baseline] = []

        # for each baseline, compute metric for all images
        for img_path_gt, img_path_pred, is_gs in zip(
            img_paths_gt, img_paths_pred, grayscales):
            # for each gt-pred pair, compute the metric
            metric = -1
            try:
                im_gt_arr = np.array(Image.open(img_path_gt))
                im_pred_arr = np.array(Image.open(img_path_pred))
                if is_gs:
                    if len(im_gt_arr.shape) == 3:
                        im_gt_arr = im_gt_arr[:, :, 0]
                    if len(im_pred_arr.shape) == 3:
                        im_pred_arr = im_pred_arr[:, :, 0]
                    if metric_name == "psnr":
                        metric = metric_fn(im_gt_arr, im_pred_arr)
                    else:
                        metric = metric_fn(im_gt_arr, im_pred_arr, multichannel=False)
                else:
                    im_gt_arr = im_gt_arr[:, :, :3]
                    im_pred_arr = im_pred_arr[:, :, :3]
                    if metric_name == "psnr":
                        metric = metric_fn(im_gt_arr, im_pred_arr)
                    else:
                        metric = metric_fn(im_gt_arr, im_pred_arr, multichannel=True)
            except:
                print(f"Calculate {metric_name} for {img_path_pred} failed")
            metrics[baseline].append(metric)
    
    return metrics


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
    result_filepath = f"metrics/results/img_{config['experiment_name']}.csv"
    with open(result_filepath, 'w', newline='') as result_file:
        resultwriter = csv.writer(result_file, delimiter=',')
        resultwriter.writerow(["baseline", "predicted image path", "psnr", "ssim"])
        for baseline, metrics in metrics_dict.items():
            # for each baseline, write all metrics
            for img_path_pred, psnr, ssim in zip(
                img_paths_pred_dict[baseline],
                metrics["psnr"],
                metrics["ssim"]):
                resultwriter.writerow([baseline, img_path_pred, psnr, ssim])
    print(f"img metrics written to {result_filepath}")