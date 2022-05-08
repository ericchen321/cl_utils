# Authors: Zhijie, Guanxiong
# Normalize all obj/ply meshes under a dir;
# Save to another dir

import trimesh
import argparse
import os
from utils import scale_to_unit_sphere


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dir', type=str, default='')
    parser.add_argument(
    '--out_dir', type=str, default='')
    args = parser.parse_args()

    for filename in os.listdir(f"{args.in_dir}/"):
        if filename.endswith((".obj", ".ply")):
            print(f"Normalizing {args.in_dir}/{filename}, output to {args.out_dir}/{filename}")
            mesh = trimesh.load(f"{args.in_dir}/{filename}")
            unit_mesh = scale_to_unit_sphere(mesh)
            unit_mesh.export(f"{args.out_dir}/{filename}")
        else:
            continue
