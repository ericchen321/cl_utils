import trimesh
import open3d as o3d
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def scale_to_unit_sphere(mesh):
    r"""
    Normalize a mesh to the unit sphere.

    :param mesh: Trimesh/Open3d mesh or point cloud
    
    Return:
        scaled mesh in its original format
    """

    if isinstance(mesh, trimesh.Scene) \
        or isinstance(mesh, trimesh.Trimesh) \
        or isinstance(mesh, trimesh.PointCloud):
        # deal with shape in Trimesh format
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()

        vertices = mesh.vertices - mesh.bounding_box.centroid
        distances = np.linalg.norm(vertices, axis=1)
        vertices /= np.max(distances)

        if isinstance(mesh, trimesh.Trimesh):
            return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
        elif isinstance(mesh, trimesh.PointCloud):
            return trimesh.PointCloud(vertices=vertices)
        else:
            raise NotImplementedError
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        # deal with shape shape in Open3d format
        centroid = mesh.get_axis_aligned_bounding_box().get_center()
        vertices = np.asarray(mesh.vertices) - centroid
        distances = np.linalg.norm(vertices, axis=1)
        vertices /= np.max(distances)
        return o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=mesh.triangles
        )
    else:
        raise NotImplementedError


def get_image_paths_and_specs(config):
    img_paths_gt = config["gt_img_paths"]
    img_paths_pred_dict = config["pred_img_paths"]
    grayscales = config["grayscales"]
    return img_paths_gt, img_paths_pred_dict, grayscales


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
            im_gt_arr = np.array(Image.open(img_path_gt))
            im_pred_arr = np.array(Image.open(img_path_pred))
            metric = None
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
            metrics[baseline].append(metric)
    
    return metrics