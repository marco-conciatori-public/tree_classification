import torch

import utils
import config
import global_constants
from models import model_utils, training, conv_2d
from data_preprocessing import data_loading, standardize_img, custom_dataset, data_augmentation


verbose = config.VERBOSE
device = utils.get_available_device(verbose=verbose)
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

img_list, tag_list = data_loading.load_data(img_folder_path=global_constants.PREPROCESSED_DATA_PATH, verbose=verbose)
print(f'Found {len(img_list)} images.')

# # apply data augmentation
# can be repeated multiple times
# new_img_list, new_tag_list = data_augmentation.random_transform_img_list(
#     img_list=img_list,
#     tag_list=tag_list,
#     # apply_probability=0.6,
# )
# img_list.extend(new_img_list)
# tag_list.extend(new_tag_list)

print(len(global_constants.TREE_INFORMATION))
model = model_utils.create_model(
    model_class_name='Conv_2d',
    input_shape=img_list[0].shape,
    num_output=len(global_constants.TREE_INFORMATION),
    model_parameters=config.MODEL_PARAMETERS,
    device=device,
)
model = conv_2d.Conv_2d(
    input_shape=img_list[0].shape,
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
temp_tensor = torch.Tensor(img_list[0])
# add batch dimension
print(f'main tensor shape: {temp_tensor.shape}')
temp_tensor = temp_tensor.unsqueeze(0)
print(f'main tensor shape: {temp_tensor.shape}')
# switch from HWC to CHW
temp_tensor = temp_tensor.permute(0, 3, 1, 2)
print(f'main tensor shape: {temp_tensor.shape}')
result = model(temp_tensor)
print(result)

train_ds = custom_dataset.Dataset_from_obs_targets(
    obs_list=img_list,
    target_list=tag_list,
    name='training_dataset',
)

train_dl = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE,
)
