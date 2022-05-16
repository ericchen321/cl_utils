# Authors: Xindong, Guanxiong
# Plot losses

from cmath import exp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import yaml


# def moving_avg(loss, window):
#     loss_avg = []
#     i = 0
#     while i < len(loss) - window + 1:
#         tmp = round(np.sum(loss[
#         i:i+window]) / window, 2)
#         loss_avg.append(tmp)
#         i += 1
#     return loss_avg


# def moving_average(x, w):
#     r"""
#     Moving average implementation adapted from
#     https://stackoverflow.com/questions/\
#     14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
#     by yatu

#     :param x: 1D array
#     :param w: window size
#     """
#     return np.convolve(x, np.ones(w), 'valid') / w
    

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
    
    print(config["loss_paths"])
    data_arr = []
    min_num_pts = -1
    for file_path in config["loss_paths"]:
        # read losses from csv
        data_pd = pd.read_csv(file_path, usecols=['Step', 'Value'])
        # smooth with EMA
        data_pd = data_pd.ewm(alpha=config["smooth"]).mean()
        data = data_pd.to_numpy()
        # track min num of steps so far
        if min_num_pts == -1:
            min_num_pts = data.shape[0]
        elif data.shape[0] < min_num_pts:
            min_num_pts = data.shape[0]
        data_arr.append(np.transpose(data, [1,0]))
    for arr_id in range(len(data_arr)):
        # we sample the last `least num of points` across all experiments
        data_arr[arr_id] = data_arr[arr_id][:, :min_num_pts]
    data_arr = np.array(data_arr)
    #print(data_arr.shape)

    # set up plot
    # size setting borrowed implementation by "Joseph" from
    # https://stackoverflow.com/questions/28816046/displaying-
    # different-images-with-actual-size-in-matplotlib-subplot
    matplotlib.rcParams['figure.dpi'] = 300
    dpi = matplotlib.rcParams['figure.dpi']
    figsize = config["width"] / float(dpi), config["height"] / float(dpi)
    fig, _ = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    # plt.plot(data[0][0], data[0][1])
    # plt.plot(data[1][0], data[1][1])
    # plt.plot(data[2][0], data[2][1])
    # plt.plot(data[3][0], data[3][1])
    for data in data_arr:
        plt.plot(data[0], data[1])
    
    # limit num of steps to show
    if config["max_num_steps"] != 0 and config["max_num_steps"] < data_arr[0, 0, -1]:
        print(f"The data contains losses over {data[0][-1]} steps;") 
        print(f"we are plotting {config['max_num_steps']} steps")
        plt.xlim(0, config["max_num_steps"])
    plt.ylim(config["loss_min"], config["loss_max"])

    plt.legend(config["exp_names"])

    if config["show_labels"]:
        plt.xlabel("Step")
        plt.ylabel("Loss")
    
    fig.tight_layout()

    fig_name = "losses_"
    for exp_name in config["exp_names"]:
        fig_name += f"+{exp_name}"
    fig.savefig(fig_name)
    plt.close(fig)