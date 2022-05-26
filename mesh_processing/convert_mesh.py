# Authors: Zhijie, Guanxiong
# Convert meshes under a dir between formats (obj/ply);
# Save outputs to another dir

import trimesh
import argparse
import os


def get_converted_filename(filename_src):
    filename_root, ext = os.path.splitext(filename_src)
    filename_dst = None
    if ext == ".obj":
        filename_dst = filename_root + ".ply"
    elif ext == ".ply":
        filename_dst = filename_root + ".obj"
    else:
        raise NotImplementedError
    return filename_dst


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
        for filename_src in os.listdir(f"{args.in_dir}/"):
            if filename_src.endswith((".obj", ".ply")):
                filename_dst = get_converted_filename(filename_src)
                print(f"Converting {args.in_dir}/{filename_src}, output to {args.out_dir}/{filename_dst}")
                mesh = trimesh.load(f"{args.in_dir}/{filename_src}")
                mesh.export(f"{args.out_dir}/{filename_dst}")
    elif args.in_mesh != "" and args.out_mesh != "":
        mesh = trimesh.load(args.in_mesh)
        mesh.export(args.out_mesh)
