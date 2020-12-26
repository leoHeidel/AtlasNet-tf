import numpy as np

import dataset



def mult_vec(mat, vec):
    v = np.ones(4)
    v[:3] = vec
    return (mat@v)[:3]

def test_random_camera():
    mat = dataset.geometry.random_camera()
    center = np.zeros(3)
    target = np.array([0,0,1])
    assert np.allclose(mult_vec(mat, center), target)

def test_camera_around_center():
    camera = dataset.geometry.camera_around_center()
    expected = np.array([
        [-1,0,0,0],
        [0,1,0,0],
        [0,0,-1,1],
        [0,0,0,1]
    ])
    assert np.allclose(camera, expected), f"Expected {expected}\nbut got {np.around(camera, 3)}"
    
    
    camera_right = dataset.geometry.camera_around_center(theta=np.pi/2)
    ux = np.array([1,0,0])
    uy = np.array([0,1,0])
    uz = np.array([0,0,1])
    
    
    """
    camera = dataset.geometry.camera_around_center(theta=np.pi/2)
    expected = np.array([
        [0,0,1,0],
        [0,1,0,0],
        [-1,0,0,1],
        [0,0,0,1]
    ])
    assert np.allclose(camera, expected), f"Expected {camera}\nbut got {np.around(camera, 3)}"
    
    camera = dataset.geometry.camera_around_center(phi=0)
    expected = np.array([
        [-1,0,0,0],
        [0,0,-1,0],
        [0,-1,0,1],
        [0,0,0,1]
    ])
    assert np.allclose(camera, expected), f"Expected {camera}\nbut got {np.around(camera, 3)}"
    """
    