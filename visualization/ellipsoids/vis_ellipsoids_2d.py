from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def overlay_ellipsoids_on_img(img_arr, ellipsoids_params, out_path):
    r"""
    Over lay ellipsoids on an image.

    :param img_arr: np array of shape (H, W, num_channels)
    :param ellipsoids_params: np array of shape (num_ellipsoids, 5)
        The 5 parameters include: 2 for center pos, 2 for radii, 1 for
        angle (in radiams). Ellipsoids' frames should 1) have their
        centers coincide with the image center, 2) have their space
        normalized to [0, 1] x [0, 1].
    :param out_path: path to saved image

    Return np array of shape (H, W, num_channels)
    """

    height, width, num_channels = None, None, None
    if len(img_arr.shape) == 2:
        height, width = img_arr.shape
        num_channels = 1
    else:
        height, width, num_channels = img_arr.shape
    num_ellipsoids = ellipsoids_params.shape[0]
    

    ellipsoids_params[:, [0, 2]] = ellipsoids_params[:, [0, 2]] * width
    ellipsoids_params[:, [1, 3]] = ellipsoids_params[:, [1, 3]] * height

    # borrowed implementation by "Joseph" from
    # https://stackoverflow.com/questions/28816046/displaying-
    # different-images-with-actual-size-in-matplotlib-subplot
    matplotlib.rcParams['figure.dpi'] = 300
    dpi = matplotlib.rcParams['figure.dpi']
    figsize = width / float(dpi), height / float(dpi)

    fig, ax = plt.subplots(figsize=figsize)
    if num_channels == 1:
        ax.imshow(img_arr, cmap='gray', vmin=0, vmax=255)    
    else:
        ax.imshow(img_arr)
    ax.axis('off')
        
    for ellip_index in range(num_ellipsoids):
        # following example from
        # https://matplotlib.org/3.1.1/gallery/shapes_and_collections/ \
        # ellipse_demo.html#sphx-glr-gallery-shapes-and-collections-ellipse-demo-py
        ctr_x = ellipsoids_params[ellip_index, 0]
        ctr_y = ellipsoids_params[ellip_index, 1]
        len_x = ellipsoids_params[ellip_index, 2]
        len_y = ellipsoids_params[ellip_index, 3]
        angle = np.degrees(ellipsoids_params[ellip_index, 4])
        ellipse = Ellipse(
            xy=(ctr_x, ctr_y),
            width=len_x,
            height=len_y,
            angle=angle)
        ax.add_artist(ellipse)
        ellipse.set_alpha(0.6)
        ellipse.set_facecolor(np.random.uniform(low=0.2, high=0.8, size=3))
    fig.canvas.draw()
    plt.subplots_adjust(
        top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches = 0)
    plt.close(fig)


def test_overlay_ellipsoids_on_img(test_name, img_path):
    im_dog = Image.open(img_path)
    im_dog_arr = np.array(im_dog) # H, W, num_channels
    num_ellipsoids = 40
    ellips = np.zeros((num_ellipsoids, 5))
    # set centers
    ellips[:, [0, 1]] = np.random.uniform(low=0, high=1, size=(num_ellipsoids, 2))
    # set axis lengths
    ellips[:, [2, 3]] = np.random.uniform(low=0, high=0.2, size=(num_ellipsoids, 2))
    # set angles
    ellips[:, -1] = np.random.uniform(low=0, high=np.pi/2)
    overlay_ellipsoids_on_img(
        im_dog_arr, ellips, f"test_overlay_ellipsoids_on_img_{test_name}.png")


if __name__ == '__main__':
    test_overlay_ellipsoids_on_img(
        "dog",
        "/media/cs533r/data_b/533R_results_1/2d_images/dog/reference.jpg")

    test_overlay_ellipsoids_on_img(
        "albert",
        "/media/cs533r/data_b/533R_results_1/2d_images/albert/reference.jpg")