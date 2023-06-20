import torch

import config
import global_constants
from models import conv_2d
from data_preprocessing import data_loading, standardize_img, data_augmentation


verbose = config.VERBOSE
img_path_list = data_loading.load_img(global_constants.INTERMEDIATE_DATA_PATH, verbose=verbose)
# # get min width and height separately
min_width, min_height = standardize_img.get_min_dimensions(img_path_list)
# if verbose >= 2:
#     print(f'Minimum width: {min_width}, minimum height: {min_height}.')
#
# str_img_path_list = []
# for img_path in img_path_list:
#     # resize images to the smallest width and height found in the dataset
#     # also save results in preprocessed data folder
#     standardize_img.resize_img(
#         img_path=img_path,
#         min_width=min_width,
#         min_height=min_height,
#     )
#     verbose = 0
# verbose = config.VERBOSE

data_list = data_loading.load_data(img_folder_path=global_constants.PREPROCESSED_DATA_PATH, verbose=verbose)
print(f'Found {len(data_list)} images.')

# # apply data augmentation
# can be repeated multiple times
# transformed_data_list = data_augmentation.random_transform_img_list(data_list, apply_probability=0.6)
# data_list.extend(transformed_data_list)

print(len(global_constants.TREE_INFORMATION))
model = conv_2d.Conv_2d(
    input_shape=(min_width, min_height, 3),
    num_output=len(global_constants.TREE_INFORMATION),
    num_conv_layers=config.NUM_CONV_LAYERS,
    dense_layers=config.DENSE_LAYERS,
    convolution_parameters=config.CONVOLUTION_PARAMETERS,
    pooling_operation=config.POOLING_OPERATION,
    pooling_parameters=config.POOLING_PARAMETERS,
    name='test_conv_2d',
    model_id=0,

)
print(model)
