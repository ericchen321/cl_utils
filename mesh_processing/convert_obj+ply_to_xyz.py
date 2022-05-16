# Author: Zhijie, Guanxiong

import numpy as np
import trimesh
import argparse
from utils import scale_to_unit_sphere


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mesh_input', type=str, default='')
    parser.add_argument(
        '--xyz_output', type=str, default='')
    parser.add_argument(
        '--num_pts', type=int, default=3000000, help="number of points to sample from the input mesh"
    )
    parser.add_argument(
        '--normalize', action='store_true', default=False, help='normalize shape before conversion or not'
    )
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh_input)

    if args.normalize:
        # normalize if asked
        mesh = scale_to_unit_sphere(mesh)

    # sample a subset of points
    samples, fid  = mesh.sample(args.num_pts, return_index=True)
    print(f"sampled {args.num_pts} points from {args.mesh_input}")
    
    # compute the barycentric coordinates of each sample
    bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[fid], points=samples)
    # interpolate vertex normals from barycentric coordinates
    interp = trimesh.unitize((mesh.vertex_normals[mesh.faces[fid]] * trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
    out_data = np.concatenate([samples, interp], axis=-1)

    np.savetxt(args.xyz_output, out_data, fmt='%1.6f')
