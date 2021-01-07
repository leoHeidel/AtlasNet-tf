import numpy as np
import trimesh

def camera_around_center(theta=0, phi=np.pi/2, dist=0):
    rot_matrix_theta = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    rot_matrix_phi = np.array([
        [1, 0, 0],
        [0, np.sin(phi), -np.cos(phi)],
        [0, np.cos(phi), np.sin(phi)]
    ])
    rot_matrix = rot_matrix_theta@rot_matrix_phi
    uz = np.array([0,0,1])
    start_pos = rot_matrix @ uz
    camera = np.zeros((4,4), dtype=np.float32)
    inv = np.array([
        [-1,0,0],
        [0,1,0],
        [0,0,-1]
    ])
    cam_rot = rot_matrix @ inv
    camera[:3,3] = -start_pos
    camera[:3,:3] = cam_rot
    camera[3,3] = 1
    return camera
    
def random_camera(phi_min=np.pi*0.2, 
                  phi_max=np.pi*0.6):
    theta = np.random.uniform(-np.pi, np.pi)
    phi = np.random.uniform(phi_min, phi_max)
    return camera_around_center(theta, phi)

def get_vertices(path):
    fuze_trimesh = trimesh.load(path)
    points = []
    if type(fuze_trimesh) == trimesh.scene.scene.Scene:
        for key in fuze_trimesh.geometry:
            points.append(fuze_trimesh.geometry[key].vertices)
    else:
        assert type(fuze_trimesh) == trimesh.base.Trimesh
        points.append(fuze_trimesh.vertices)
    return np.concatenate(points)

def get_area(triangles):
    v1 = triangles[:,1] - triangles[:,0]
    v2 = triangles[:,2] - triangles[:,1]
    area = np.linalg.norm(np.cross(v1,v2), axis=1)
    return area

def sample_points(path, nb=10000):
    fuze_trimesh = trimesh.load(path)
    triangles = []
    if type(fuze_trimesh) == trimesh.scene.scene.Scene:
        for key in fuze_trimesh.geometry:
            triangles.append(fuze_trimesh.geometry[key].triangles)
    else:
        assert type(fuze_trimesh) == trimesh.base.Trimesh
        triangles.append(fuze_trimesh.triangles)
    triangles = np.concatenate(triangles)
    area = get_area(triangles)
    proba = area / np.sum(area)
    samples = np.random.choice(len(proba), size=nb, p=proba)
    coord = np.random.uniform(size=(nb, 3, 1))
    coord = coord / np.sum(coord, axis=1, keepdims=True)
    basis = np.take(triangles, samples, axis=0)
    points = np.sum(coord*basis, axis=1)
    return points

def mult_vec(mat, vec):
    v = np.ones((len(vec),4))
    v[:,:3] = vec
    return (v@mat)[:,:3]
