import numpy as np


def camera_around_center(theta=0, phi=np.pi/2, dist=0):
    rot_matrix_theta = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    rot_matrix_phi = np.array([
        [1, 0, 0],
        [0, np.sin(phi), np.cos(phi)],
        [0, -np.cos(phi), np.sin(phi)]
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
    camera[:3,3] = - inv @ uz
    camera[:3,:3] = inv @ rot_matrix.T
    camera[3,3] = 1
    return camera
    
def random_camera(phi_min=np.pi*0.3, 
                  phi_max=np.pi*1.2):
    theta = np.random.uniform(-np.pi, np.pi)
    phi = np.random.uniform(phi_min, phi_max)
    return camera_around_center(theta, phi)