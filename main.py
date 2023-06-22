import torch

import utils
import config
import global_constants
from models import model_utils, training, conv_2d
from data_preprocessing import data_loading, standardize_img, custom_dataset, data_augmentation


verbose = config.VERBOSE
device = utils.get_available_device(verbose=verbose)

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
