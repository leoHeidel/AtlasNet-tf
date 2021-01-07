import tempfile

import numpy as np

import dataset

def test_render():
    path = "dataset/test/test_data/873f4d2e92681d12709eb7790ef48e0c/models/model_normalized.obj"
    mat = np.array([[ 1.0000000e+00,  8.6595606e-17, -8.6595606e-17, -8.6595606e-17],
                    [ 0.0000000e+00,  7.0710677e-01,  7.0710677e-01,  7.0710677e-01],
                    [ 1.2246469e-16, -7.0710677e-01,  7.0710677e-01,  7.0710677e-01],
                    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
    dataset.rendering.render(path, mat)
    #second rendering might fail
    dataset.rendering.render(path, mat)
