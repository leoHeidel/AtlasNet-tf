import dataset

planes_input = "/Data/leo/download/ShapeNetCore.v2/02691156/"
planes_output = "data/planes"

dataset.dataset.make_dataset(planes_input, planes_output, size=128, 
                             nb_points=10000, number_models=1000000, 
                             nb_samples_per_model=50)
