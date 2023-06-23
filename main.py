import torch

import utils
import config
import global_constants
from models import model_utils, training
from data_preprocessing import data_loading, standardize_img, custom_dataset, data_augmentation


verbose = config.VERBOSE
device = utils.get_available_device(verbose=verbose)

model = model_utils.create_model(
    model_class_name='Conv_2d',
    input_shape=img_list[0].shape,
    num_output=len(global_constants.TREE_INFORMATION),
    model_parameters=config.MODEL_PARAMETERS,
    device=device,
    name='test_conv_2d',
    verbose=verbose,
)
# print(model)
# temp_tensor = torch.Tensor(img_list[0])
# # add batch dimension
# print(f'main tensor shape: {temp_tensor.shape}')
# temp_tensor = temp_tensor.unsqueeze(0)
# print(f'main tensor shape: {temp_tensor.shape}')
# # switch from HWC to CHW
# temp_tensor = temp_tensor.permute(0, 3, 1, 2)
# print(f'main tensor shape: {temp_tensor.shape}')
# temp_tensor = temp_tensor.to(device)
# print(f'main tensor shape: {temp_tensor.shape}')
# result = model(temp_tensor)
# print(result)

training_history = training.train(
    model=model,
    train_data=train_dl,
    epochs=config.EPOCHS,
    learning_rate=config.LEARNING_RATE,
    loss_function_name=config.LOSS_FUNCTION_NAME,
    optimizer_name=config.OPTIMIZER_NAME,
    device=device,
    verbose=verbose,
    save_model=True,
    save_path=global_constants.MODEL_OUTPUT_DIR,
    # metrics=config.METRICS,
)

print('training_history:')
print(training_history)
