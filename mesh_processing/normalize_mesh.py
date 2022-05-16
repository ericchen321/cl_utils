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
    parser.add_argument(
        '--in_mesh', type=str, default='')
    parser.add_argument(
        '--out_mesh', type=str, default='')
    args = parser.parse_args()

    if args.in_dir != "" and args.out_dir != "":
        for filename in os.listdir(f"{args.in_dir}/"):
            if filename.endswith((".obj", ".ply")):
                print(f"Normalizing {args.in_dir}/{filename}, output to {args.out_dir}/{filename}")
                mesh = trimesh.load(f"{args.in_dir}/{filename}")
                unit_mesh = scale_to_unit_sphere(mesh)
                unit_mesh.export(f"{args.out_dir}/{filename}")
            else:
                continue
    elif args.in_mesh != "" and args.out_mesh != "":
        mesh = trimesh.load(args.in_mesh)
        unit_mesh = scale_to_unit_sphere(mesh)
        unit_mesh.export(args.out_mesh)
