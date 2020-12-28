import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender

import dataset

def render(obj_path, camera_mat, return_depth=False):
    fuze_trimesh = trimesh.load(obj_path)
    scene = pyrender.Scene.from_trimesh_scene(fuze_trimesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    s = np.sqrt(2)/2
    scene.add(camera, pose=camera_mat)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi/16.0)
    scene.add(light, pose=camera_mat)
    r = pyrender.OffscreenRenderer(512, 512)
    color, depth = r.render(scene)
    if return_depth:
        return color,depth
    return color