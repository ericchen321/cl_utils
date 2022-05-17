# Author: Chunjin Song
# Note: for the threshold, we're using values used in other papers,
# but we might want to pick a different one

import os, typing
import open3d as o3d
import numpy as np

def calculate_fscore(gt: o3d.geometry.PointCloud, pr: o3d.geometry.PointCloud, th: float = 0.01) -> typing.Tuple[
    float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall


mesh_src = './normalized_shapes/bimba_gt_normalized.ply'
mesh_pred1 = './normalized_shapes/bimba_instantngp_normalized.ply'
mesh_pred2 = './normalized_shapes/bimba_lslayer_normalized.ply'
mesh_pred3 = './normalized_shapes/bimba_FFN_normalized.ply'
out_dir = './fcore'

src_path_lst = [mesh_src, mesh_src, mesh_src]
pred_path_lst = [mesh_pred1, mesh_pred2, mesh_pred3]

CUBE_SIDE_LEN = 1.0

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

threshold_list = [CUBE_SIDE_LEN / 200, CUBE_SIDE_LEN / 100,
                      CUBE_SIDE_LEN / 50, CUBE_SIDE_LEN / 20,
                      CUBE_SIDE_LEN / 10, CUBE_SIDE_LEN / 5]

os.makedirs(out_dir, exist_ok=True)
f_f = open(os.path.join(out_dir, "fscore.txt"), "w")
f_p = open(os.path.join(out_dir, "precision.txt"), "w")
f_r = open(os.path.join(out_dir, "recall.txt"), "w")
for src_path, prep_path in zip(src_path_lst, pred_path_lst):
    gt = np.asarray(o3d.io.read_triangle_mesh(src_path).vertices)
    pr = np.asarray(o3d.io.read_triangle_mesh(prep_path).vertices)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt)

    pcd_pr = o3d.geometry.PointCloud()
    pcd_pr.points = o3d.utility.Vector3dVector(pr)

    f_f.write(" " + str(prep_path) + "\n")
    f_p.write(" " + str(prep_path) + "\n")
    f_r.write(" " + str(prep_path) + "\n")

    for th in threshold_list:
        f, p, r = calculate_fscore(pcd_gt, pcd_pr, th=th)
        print(prep_path, f'fscore: {str(f)}, precision: {str(p)}, recall: {str(r)}')


        f_f.write(" " + str(f)  + "\n")
        f_p.write(" " + str(p) + "\n")
        f_r.write(" " + str(r) + "\n")

    f_f.write("\n")
    f_p.write("\n")
    f_r.write("\n")

f_f.close()
f_p.close()
f_r.close()
