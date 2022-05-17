# Author: Chunjin Song
# Note: larger IoU leads to more accurate result, but is slower
# to compute

import numpy as np
import open3d as o3d
from joblib import Parallel, delayed


def iou_pymesh(mesh_src, mesh_pred, scale):
    mesh1 = o3d.io.read_triangle_mesh(mesh_src)
    grid1 = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh1, 0.1/scale, [-1,-1,-1], [1,1,1])
    vertices_grid1 = np.asarray([x.grid_index for x in grid1.get_voxels()])

    mesh2 = o3d.io.read_triangle_mesh(mesh_pred)
    grid2 = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh2, 0.1/scale,[-1,-1,-1], [1,1,1])
    vertices_grid2 = np.asarray([x.grid_index for x in grid2.get_voxels()])

    dim = 20 * scale
    v1 = np.zeros([dim, dim, dim])
    v1[vertices_grid1[:, 0], vertices_grid1[:, 1], vertices_grid1[:, 2]] = 1

    v2 = np.zeros([dim, dim, dim])
    v2[vertices_grid2[:, 0], vertices_grid2[:, 1], vertices_grid2[:, 2]] = 1

    intersection = np.sum(np.logical_and(v1, v2))
    union = np.sum(np.logical_or(v1, v2))
    print(float(intersection) / union, mesh_pred)
    return [float(intersection) / union, mesh_pred]


mesh_src = './normalized_shapes/bimba_gt_normalized.ply'
mesh_pred1 = './normalized_shapes/bimba_instantngp_normalized.ply'
mesh_pred2 = './normalized_shapes/bimba_lslayer_normalized.ply'
mesh_pred3 = './normalized_shapes/bimba_FFN_normalized.ply'


src_path_lst = [mesh_src, mesh_src, mesh_src]
pred_path_lst = [mesh_pred1, mesh_pred2, mesh_pred3]
scale = 1
scales = [scale, scale, scale]
with Parallel(n_jobs=20) as parallel:
    result_lst = parallel(delayed(iou_pymesh)
                             (src_path, pred_path, scale)
                             for src_path, pred_path, scale in
                             zip(src_path_lst, pred_path_lst, scales))

for result in result_lst:
    print(result[0], result[1])

# iou_vals = np.asarray([result[0] for result in result_lst], dtype=np.float32)
# iou_pred = [result[1] for result in result_lst]
# print(iou_vals)
# print(iou_pred)


# def iou_pymesh(mesh_src, mesh_pred, dim=FLAGS.dim):
#     try:
#         mesh1 = pymesh.load_mesh(mesh_src)
#         grid1 = pymesh.VoxelGrid(2./dim)
#         grid1.insert_mesh(mesh1)
#         grid1.create_grid()
#
#         ind1 = ((grid1.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
#         v1 = np.zeros([dim, dim, dim])
#         v1[ind1[:,0], ind1[:,1], ind1[:,2]] = 1
#
#
#         mesh2 = pymesh.load_mesh(mesh_pred)
#         grid2 = pymesh.VoxelGrid(2./dim)
#         grid2.insert_mesh(mesh2)
#         grid2.create_grid()
#
#         ind2 = ((grid2.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
#         v2 = np.zeros([dim, dim, dim])
#         v2[ind2[:,0], ind2[:,1], ind2[:,2]] = 1
#
#         intersection = np.sum(np.logical_and(v1, v2))
#         union = np.sum(np.logical_or(v1, v2))
#         return [float(intersection) / union, mesh_pred]
#     except:
#         print("error mesh {} / {}".format(mesh_src, mesh_pred))
