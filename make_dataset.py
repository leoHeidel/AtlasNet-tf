import dataset

planes_input = "/Data/leo/download/ShapeNetCore.v2/*/"
planes_output = "data/all_256"

dataset.dataset.make_dataset(planes_input, planes_output, size=256, 
                             nb_points=10000, number_models=1000000, 
                             nb_samples_per_model=1, overwrite=False,
                             fast_skip=True)
