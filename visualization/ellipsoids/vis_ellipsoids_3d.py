# Author: Chunjin

import trimesh
import pyrender
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

scene = pyrender.Scene()
# scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[1, 1, 1])
# camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
# light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
# light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
#                            innerConeAngle=np.pi/16.0,
#                            outerConeAngle=np.pi/6.0)
# s = np.sqrt(2)/2
# camera_pose = np.array([
#     [0.0, -s,   s,   0.3],
#     [1.0,  0.0, 0.0, 0.0],
#     [0.0,  s,   s,   0.35],
#     [0.0,  0.0, 0.0, 1.0],
#  ])
#
# scene.add(light, pose=camera_pose)

center_circle = np.array([[-0.2,0.2, 0.2],
                          [-0.2,0.8, 0.2],
                          [-0.2, 0.8, -0.3],
                          [-0.5, 0.3, 0.4],
                          [0.8, -0.5, 0.5],
                          [0.7, 0.2, 0.8]])
radius = 0.1
obj_color = [0.5, 0.9, 0.9, 0.2]
alpha = 0.7
circle_color = [[0.0, 0.0, 0.9, alpha],
                [0.0, 0.9, 0.3, alpha],
                [0.4, 0.0, 0.4, alpha],
                [0.3, 0.5, 0.8, alpha],
                [0.5, 0.5, 0.1, alpha],
                [0.5, 0.1, 0.6, alpha],
                [0.4, 1.0, 1.0, alpha],
                [0.7, 1.0, 0.0, alpha],
                [0.8, 1.0, 0.0, alpha],
                [0.8, 0.4, 0.0, alpha],
                [0.9, 0.0, 0.7 , alpha]]
path = './beardman_normalized.ply'
tri_mesh = trimesh.load_mesh(path)
tri_mesh = trimesh.Trimesh(tri_mesh.vertices, tri_mesh.faces, vertex_colors=obj_color)
mesh = pyrender.Mesh.from_trimesh(tri_mesh)
scene.add(mesh, pose=  np.eye(4))

# sm = trimesh.creation.uv_sphere(radius=radius)
sm = trimesh.creation.icosphere(subdivisions=4, radius=radius)

for i in range(len(center_circle)):
    sm.vertices += center_circle[i]
    vertex_colors = circle_color[random.randint(0, len(circle_color)-1)]
    tri_mesh = trimesh.Trimesh(sm.vertices, sm.faces, vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene.add(mesh, pose=  np.eye(4))

pyrender.Viewer(scene, use_raymond_lighting=True)

# c = 2**-0.5
# scene.add(camera, pose=[[ 1,  0,  0,  0],
#                         [ 0,  c, -c, -2],
#                         [ 0,  c,  c,  2],
#                         [ 0,  0,  0,  1]])
#
# # render scene
# r = pyrender.OffscreenRenderer(512, 512)
# color, depth = r.render(scene)
#
# plt.figure()
# plt.subplot(1,2,1)
# plt.axis('off')
# plt.imshow(color)
# plt.subplot(1,2,2)
# plt.axis('off')
# plt.imshow(depth, cmap=plt.cm.gray_r)
# plt.show()
#
