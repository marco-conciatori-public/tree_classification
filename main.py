import cv2
import config
import global_constants
from data_preprocessing import data_loading, standardize_img, data_augmentation


verbose = config.VERBOSE
img_list = data_loading.load_img(global_constants.INTERMEDIATE_DATA_PATH, verbose=verbose)
# get min width and height separately
min_width, min_height = standardize_img.get_min_dimensions(img_list)
if verbose >= 2:
    print(f'Minimum width: {min_width}, minimum height: {min_height}.')

for img_path in img_list:
    # resize images to the smallest width and height found in the dataset
    # also save results in preprocessed data folder
    standardize_img.resize_img(img_path, min_width, min_height)

data_list = data_loading.load_data(img_path_list=img_list, verbose=verbose)
print(f'Found {len(data_list)} images.')

transformed_data_list = data_augmentation.random_transform_img_list(data_list, apply_probability=0.6)
