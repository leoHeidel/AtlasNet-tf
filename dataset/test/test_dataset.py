import tempfile

import numpy as np

import dataset

def test_make_read_dataset():
    path = "dataset/test/test_data/"
    with tempfile.TemporaryDirectory() as tmp_dirname:
        #Making
        dataset.dataset.make_dataset(path, tmp_dirname, nb_samples_per_model=2)
        
        #Reading
        tf_dataset = dataset.dataset.read_dataset(tmp_dirname)
        for x,y in tf_dataset:
            pass