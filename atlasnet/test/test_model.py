import atlasnet

import tensorflow.keras as keras


def test_full_model():
    nb_mlps = 2
    features_model = atlasnet.model.get_features_model()
    mlp_models = [atlasnet.model.get_mlp_model_small() for i in range(nb_mlps)]
    full_model = atlasnet.model.get_full_model(features_model, mlp_models)
    full_model.compile(keras.optimizers.Adam(lr=0.0001), loss=atlasnet.model.chamfer_loss)