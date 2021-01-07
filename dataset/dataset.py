import glob
import pickle
import os

import numpy as np
import OpenGL
import PIL.Image as Image
import tensorflow as tf
import tqdm 
import trimesh

import dataset

def make_dataset(input_path, output_path, size=128, nb_points=10000, number_models=None, nb_samples_per_model=20, gl_tries=5):
    dataset.utils.make_dir(output_path)
    objects_path = os.path.join(input_path, "*/models/*.obj")
    objects_path = glob.glob(objects_path)
    if number_models is not None:
        objects_path = objects_path[:number_models]
    for path in tqdm.tqdm(objects_path):
        name = path.split(os.path.sep)[-3]
        object_dir = os.path.join(output_path, name)
        dataset.utils.make_dir(object_dir)
        pts_path = os.path.join(object_dir, "pts.pkl")
        pts = dataset.geometry.sample_points(path, nb=nb_points)
        pts = np.array(pts, dtype=np.float32)
        with open(pts_path, 'wb') as handle:
            pickle.dump(pts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        for i in range(nb_samples_per_model):
            render_name = f"render_{str(i).zfill(5)}"
            mat_name = f"mat_{str(i).zfill(5)}"
            image_path = os.path.join(object_dir, render_name + ".jpg") 
            mat_path = os.path.join(object_dir, render_name + ".pkl") 
            mat = dataset.geometry.random_camera()
            
            for i in range(gl_tries):
                try:
                    color = dataset.rendering.render(path, mat, im_size=size)
                    break
                except OpenGL.error.GLError:
                    print(f"GL Error occured, try {i}, trying again.")
            else:
                color = dataset.rendering.render(path, mat, im_size=size)
            
            im = Image.fromarray(color)
            im.save(image_path)
            with open(mat_path, 'wb') as handle:
                pickle.dump(mat, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                
def load_example(image_path):
    number = image_path[-9:-4]
    base_dir = image_path[:-16]
    mat_path = os.path.join(base_dir, f"render_{number}.pkl")
    with open(mat_path, 'rb') as handle:
        mat = pickle.load(handle)
    pts_path = os.path.join(base_dir, "pts.pkl")
    with open(pts_path, 'rb') as handle:
        pts = pickle.load(handle)
    pts = dataset.geometry.mult_vec(mat, pts)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255
    return img, pts

def read_dataset(path, batch_size=4):
    images_paths = os.path.join(path, "*", "*.jpg")
    images_paths = glob.glob(images_paths)
    def gen():
        np.random.shuffle(images_paths)
        for image_path in images_paths:
            yield load_example(image_path)
    tf_dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32))
    tf_dataset = tf_dataset.batch(batch_size)
    return tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
