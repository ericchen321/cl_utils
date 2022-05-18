from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import yaml


def setup_fig(height, width):
    # borrowed implementation by "Joseph" from
    # https://stackoverflow.com/questions/28816046/displaying-
    # different-images-with-actual-size-in-matplotlib-subplot
    matplotlib.rcParams['figure.dpi'] = 300
    dpi = matplotlib.rcParams['figure.dpi']
    figsize = width / float(dpi), height / float(dpi)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def make_binary_image(bin, r=95, g=137, b=210):
    r"""
    Generate binary image from a binary mask. mask should have
    filled regions being True's and unfilled regions being False's.
    """
    # use blue and white mix
    row_ids, col_inds = np.nonzero(bin.astype(np.uint8))
    bin = (~bin).astype(np.float32)
    img_bin = np.repeat(np.expand_dims(bin, 2), 3, axis=2) * 255
    img_bin[row_ids, col_inds, 0] = r
    img_bin[row_ids, col_inds, 1] = g
    img_bin[row_ids, col_inds, 2] = b
    return img_bin.astype(np.uint8)


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

    # get image stats
    img =  np.asarray(Image.open(config["img_path"]).convert('L'), dtype=np.uint8)
    assert img.shape[0] == img.shape[1]
    img_size = img.shape[0]
    # img = img[0:img_size, 170:170+img_size]
    print(f"image shape: {img.shape}")

    # binarize
    fig, ax = None, None
    if config["size"] != -1:
        fig, ax = setup_fig(config["size"], config["size"])
    else:
        fig, ax = setup_fig(img_size, img_size)
    bin = img > 0.0
    img_bin = make_binary_image(
        bin, r=config["bin_r"], g=config["bin_g"], b=config["bin_b"])
    ax.imshow(img_bin)
    ax.axis("off")
    plt.savefig(
        f"visualization/sse_diagram/plots/bin_{config['shape_name']}.png",
        bbox_inches='tight',
        pad_inches = 0)
    print(f"saved fig")
    plt.close(fig)

    # discretize into neighborhoods:
    neighborhood_size = config["neighborhood_size"]
    # expand each neighborhood into a 1D vector
    neighborhoods = bin.astype(np.float32).reshape(
        int(img_size/neighborhood_size),
        neighborhood_size,
        int(img_size/neighborhood_size),
        neighborhood_size).transpose(0, 2, 1, 3)
    # print(f"neighborhoods shape: {neighborhoods.shape}")
    neighborhoods_flat = neighborhoods.reshape(
        -1,
        neighborhood_size*neighborhood_size)
    # print(f"neighborhoods_flat shape: {neighborhoods_flat.shape}")
    # compute mean
    means = np.expand_dims(
        np.mean(neighborhoods_flat, 1), 1)
    # print(f"means shape: {means.shape}")
    # compute SSE
    sses = np.diag(np.matmul(
        (neighborhoods_flat - means),
        (neighborhoods_flat - means).transpose(1, 0)
    )).reshape(
        int(img_size/neighborhood_size),
        int(img_size/neighborhood_size)
    )
    # print(f"sses shape: {sses.shape}")

    # visualize sses as a heatmap
    if config["size"] != -1:
        fig, ax = setup_fig(config["size"], 1.2*config["size"])
    else:
        fig, ax = setup_fig(img_size, 1.2*img_size)
    hmap = sses / np.amax(np.abs(sses))
    ax = sns.heatmap(hmap, cmap="YlGnBu")
    ax.axis('off')
    if config["show_title"]:
        fig.suptitle(f"Per {neighborhood_size}x{neighborhood_size} Neighborhood SSE")
    plt.subplots_adjust(
        top = 0.95, bottom = 0.05, right = 0.9, left = 0.05, hspace = 0, wspace = 0)
    plt.savefig(
        f"visualization/sse_diagram/plots/sse_{config['shape_name']}.png")
    plt.close(fig)
