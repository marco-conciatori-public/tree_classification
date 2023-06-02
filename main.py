import config
import global_constants
from data_preprocessing import data_loading, standardize_img


verbose = config.VERBOSE
img_list = data_loading.load_img(global_constants.DATA_PATH, verbose=verbose)
# get min width and height separately
min_width, min_height = standardize_img.get_min_dimensions(img_list)
if verbose >= 2:
    print(f'Minimum width: {min_width}, minimum height: {min_height}.')

for img_path in img_list:
    # resize images to the smallest width and height found in the dataset
    standardize_img.resize_img(img_path, min_width, min_height)
