import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import trimesh

def render(obj_path, camera_mat, return_depth=False, im_size=128):
    fuze_trimesh = trimesh.load(obj_path)
    if type(fuze_trimesh) == trimesh.base.Trimesh:
        m = pyrender.Mesh.from_trimesh(fuze_trimesh)
        scene = pyrender.Scene()
        scene.add_node(pyrender.Node(mesh=m))
    else:
        assert type(fuze_trimesh) == trimesh.scene.scene.Scene, "Unrognized file"
        scene = pyrender.Scene.from_trimesh_scene(fuze_trimesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    s = np.sqrt(2)/2
    scene.add(camera, pose=camera_mat)
    light = pyrender.SpotLight(color=np.ones(3), intensity=4.0,
                               innerConeAngle=np.pi/16.0)
    scene.add(light, pose=camera_mat)
    light = pyrender.SpotLight(color=np.ones(3), intensity=6.0,
                               innerConeAngle=0.2*np.pi)
    light_pose = np.array([
        [0,1,0,0],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1]
    ], dtype=np.float32)
    scene.add(light, pose=light_pose)
    r = pyrender.OffscreenRenderer(im_size, im_size)
    color, depth = r.render(scene)
    if return_depth:
        return color,depth
    return color
