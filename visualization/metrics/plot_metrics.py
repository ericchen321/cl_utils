# Author: Guanxiong

import matplotlib
import matplotlib.pyplot as plt
import argparse
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

    psnrs, ssims = [], []
    for img_path in config["img_paths"][1:]:
        im_pred_arr = np.array(Image.open(img_path))
        psnrs.append(peak_signal_noise_ratio(im_gt_arr, im_pred_arr))
        ssims.append(structural_similarity(im_gt_arr, im_pred_arr, multichannel=True))

    # size setting borrowed implementation by "Joseph" from
    # https://stackoverflow.com/questions/28816046/displaying-
    # different-images-with-actual-size-in-matplotlib-subplot
    matplotlib.rcParams['figure.dpi'] = 300
    dpi = matplotlib.rcParams['figure.dpi']
    figsize = config["width"] / float(dpi), config["height"] / float(dpi)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)    

    if config["plot_type"] == "bar":
        # implmentation from
        # https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        # bar_width = 0.25
        # pred_ids = np.arange(len(config["img_paths"])-1)
        # ax.bar(pred_ids + 0.00, psnrs, color = 'b', width=bar_width)
        # ax.bar(pred_ids + bar_width, ssims, color = 'r', width=bar_width)

        # ax.set_xticks(pred_ids+bar_width/2)
        # ax.set_xticklabels(args.exp_names[1:])
        # ax.set_ylabel("Metric")

        # for bars in ax.containers:
        #     ax.bar_label(bars)
        
        # ax.legend(labels=["PSNR", "SSIM"], loc="upper right")

        # plt.subplots_adjust(
        #     top = 0.95, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        # fig.savefig("bars.png")
        bar_width = config["bar_width"]
        pred_ids = np.arange(len(config["img_paths"])-1)
        axes[0].bar(pred_ids, psnrs, color=config["color_psnr"], width=bar_width)
        axes[1].bar(pred_ids, ssims, color=config["color_ssim"], width=bar_width)
        if config["show_y_label"]:
            axes[0].set_ylabel("PSNR")
            axes[1].set_ylabel("SSIM")

        # add x ticks
        for metric_id in range(2):
            axes[metric_id].set_xticks(pred_ids)
            axes[metric_id].set_xticklabels(config["hparams"])

        # add x labels
        if config["show_x_label"]:
            axes[1].set_xlabel(config["hparam_name"])
        
        # add bar labels
        for metric_id in range(2):
            for bars in axes[metric_id].containers:
                if metric_id == 0:
                    axes[metric_id].bar_label(bars, fmt='%.2f')
                    axes[metric_id].set_ylim(
                        bottom=config["y_min_psnr"],
                        top=config["y_max_psnr"])
                else:
                    axes[metric_id].bar_label(bars, fmt='%.4f')
                    axes[metric_id].set_ylim(
                        bottom=config["y_min_ssim"],
                        top=config["y_max_ssim"])

        plt.subplots_adjust(
            top = 0.95, bottom = 0.08, right = 0.95, left = 0.1, hspace = 0.05, wspace = 0)
    
    elif config["plot_type"] == "scatter":
        # plot and add y labels
        for metric_id in range(2):
            c = "b" if metric_id == 0 else "r"
            val = psnrs if metric_id == 0 else ssims
            axes[metric_id].scatter(
                config["hparams"],
                val,
                c=config["color_psnr"],
                s=8,
                marker='o',
                linewidths=2.5)
            axes[metric_id].plot(
                config["hparams"],
                val,
                c=["color_ssim"])
            if config["show_y_label"]:
                y_label = "PSNR" if metric_id == 0 else "SSIM"
                axes[metric_id].set_ylabel(y_label)

        # add point labels
        for i, hparam in enumerate(config["hparams"]):
            axes[0].annotate(f"{psnrs[i]: .2f}", (hparam, psnrs[i]))
            axes[1].annotate(f"{ssims[i]: .4f}", (hparam, ssims[i]))
        for metric_id in range(2):
            axes[metric_id].set_xlim(right=config["x_max"])
            if metric_id == 0:
                axes[metric_id].set_ylim(
                    bottom=config["y_min_psnr"],
                    top=config["y_max_psnr"])
            else:
                axes[metric_id].set_ylim(
                    bottom=config["y_min_ssim"],
                    top=config["y_max_ssim"])

        # add x labels
        if config["show_x_label"]:
            axes[1].set_xlabel(config["hparam_name"])
        
        plt.subplots_adjust(
            top = 0.95, bottom = 0.08, right = 0.95, left = 0.1, hspace = 0.02, wspace = 0)
    
    else:
        raise NotImplementedError
    
    fig.savefig(f"visualization/metrics/plots/{config['plot_filepath']}")
    