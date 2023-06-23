import utils
import config
import global_constants
from models import model_utils, training, evaluation
from data_preprocessing import get_ready_data


verbose = config.VERBOSE
device = utils.get_available_device(verbose=verbose)

train_dl, val_dl, test_dl, img_shape = get_ready_data.get_data(
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE,
    train_val_test_proportions=config.TRAIN_VAL_TEST_PROPORTIONS,
    tolerance=config.TOLERANCE,
    augment_data=config.DATA_AUGMENTATION_PROPORTION,
    verbose=verbose,
)

model = model_utils.create_model(
    model_class_name='Conv_2d',
    input_shape=img_shape,
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
    training_data=train_dl,
    validation_data=val_dl,
    epochs=config.EPOCHS,
    learning_rate=config.LEARNING_RATE,
    loss_function_name=config.LOSS_FUNCTION_NAME,
    optimizer_name=config.OPTIMIZER_NAME,
    device=device,
    verbose=verbose,
    save_model=True,
    save_path=global_constants.MODEL_OUTPUT_DIR,
    metrics=config.METRICS,
)
print('training_history:')
print(training_history)

# TODO: test metric, reactivate save_model and create a script for loading and testing models

test_loss, metric_evaluations = evaluation.eval(
    model=model,
    test_data=test_dl,
    loss_function_name=config.LOSS_FUNCTION_NAME,
    device=device,
    metrics=config.METRICS,
    verbose=verbose,
)
print(f'test_loss: {test_loss}')
print(f'test_metric_evaluations: {metric_evaluations}')
