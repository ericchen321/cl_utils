# Authors: Zhijie, Guanxiong
# Normalize all obj/ply meshes under a dir;
# Save to another dir

import trimesh
import yaml
import argparse
from utils.utils import scale_to_unit_sphere


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, help='path to yaml config file', required=True)
    args = p.parse_args()

    config = None
    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for mesh_in_path, mesh_out_path in zip(config["mesh_in_paths"], config["mesh_out_paths"]):
        mesh = trimesh.load(mesh_in_path)
        unit_mesh = scale_to_unit_sphere(mesh)
        unit_mesh.export(mesh_out_path)
        print(f"Normalized {mesh_in_path}, saved to {mesh_out_path}")
