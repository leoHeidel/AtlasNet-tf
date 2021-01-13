from sklearn.neighbors import KDTree
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

def get_features_model(trainable=True, hidden_size=1024, pretrained_model=None):
    if pretrained_model == None:
        pretrained = keras.applications.Xception(include_top=False)
    pretrained.trainable=trainable
    return keras.models.Sequential([
        pretrained,
        keras.layers.Dense(hidden_size),
        keras.layers.Flatten()
    ])

def get_mlp_model_article(hidden_sizes=[512, 256, 7500], activation=None):
    if activation is None:
        activation = keras.layers.Activation(keras.activations.relu) 
    layers = []
    for hidden_size in hidden_sizes[:-1]: 
        layers.append(keras.layers.Dense(hidden_size))
        layers.append(activation)
    
    layers.append(keras.layers.Dense(hidden_sizes[-1], activation="tanh"))
    layers.append(keras.layers.Dense(3))
    
    return keras.models.Sequential(layers)
    
def get_mlp_model_small(hidden_sizes=[512, 256, 256], activation=None):
    if activation is None:
        activation = keras.layers.Activation(keras.activations.relu) 
    layers = []
    for hidden_size in hidden_sizes: 
        layers.append(keras.layers.Dense(hidden_size))
        layers.append(activation)
    
    layers.append(keras.layers.Dense(3))
    
    return keras.models.Sequential(layers)

def get_full_model(features_model, mlp_models, hidden=1024, activation=None, im_size=128):
    """
    Assamble the atlas net model
    each input_coord is expected of shape [batch, n_points, 3]
    """
    if activation is None:
        activation = keras.layers.Activation(keras.activations.relu)
        
    input_image = keras.layers.Input((im_size,im_size,3))
    inputs_coord = [keras.layers.Input((None,2)) for _ in mlp_models]
    features = features_model(input_image)
    outputs = []
    for mlp_model, coord in zip(mlp_models, inputs_coord):
        x = keras.layers.Dense(hidden)(features)
        coord = keras.layers.Dense(hidden, use_bias=False)(coord)
        x = activation(x[:,tf.newaxis,:]+coord)
        output = mlp_model(x)
        outputs.append(output)
    output = tf.concat(outputs, axis=1)
    return keras.models.Model((input_image, *inputs_coord), output)
    

def chamfer_loss(y_true, y_pred):
    difference = (
        tf.expand_dims(y_true, axis=-2) -
        tf.expand_dims(y_pred, axis=-3))
    # Calculate the square distances between each two points: |ai - bj|^2.
    square_distances = tf.einsum("...i,...i->...", difference, difference)

    minimum_square_distance_a_to_b = tf.reduce_min(
        input_tensor=square_distances, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_min(
        input_tensor=square_distances, axis=-2)

    return (
        tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
        tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))

def non_tf_code(X,Y):
    X = X
    Y = Y
    kdt1 = KDTree(X, leaf_size=30, metric='euclidean')
    nb1 = kdt1.query(Y, k=1, return_distance=False)
    kdt2 = KDTree(Y, leaf_size=30, metric='euclidean')
    nb2 = kdt2.query(X, k=1, return_distance=False)
    return nb1.astype(np.int32), nb2.astype(np.int32)

@tf.function
def chamfer_loss_fast(y_true, y_pred):
    res = 0.
    for k in tf.range(tf.shape(y_true)[0]):
        X = y_true[k]
        Y = y_pred[k]
        nb1,nb2 = tf.numpy_function(non_tf_code, [X,Y], (tf.int32, tf.int32))
        diff1 = Y - tf.gather(X,nb1[:,0])
        diff2 = X - tf.gather(Y,nb2[:,0])
        res = tf.reduce_mean(diff1*diff1) + tf.reduce_mean(diff2*diff2) + res
    return res
