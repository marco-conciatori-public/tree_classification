import torchvision.transforms.functional as tf

import utils
import config
import global_constants
from models import training, evaluation, model_utils
from data_preprocessing import get_ready_data


verbose = 2
num_classes = len(global_constants.TREE_INFORMATION)
augment_data = config.DATA_AUGMENTATION_PROPORTION
device = utils.get_available_device(verbose=verbose)
# warning: case-sensitive names
# regnet model
model_version = 'regnet_y_1_6gf'  # small
# weights_name = 'RegNet_Y_1_6GF_Weights.IMAGENET1K_V1'
# weights_name = 'RegNet_Y_1_6GF_Weights.DEFAULT'
weights_name = None
# model_version = 'regnet_y_128gf'  # big
# weights_name = 'RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1'
# freeze_layers = True
freeze_layers = False

# load model
model = model_utils.get_torchvision_model(
    model_name=model_version,
    weights_name=weights_name,
    freeze_layers=freeze_layers,
    device=device,
    training=True,
    num_classes=num_classes,
    verbose=verbose,
)
# print(f'model:\n{model}')
custom_transforms, resize_in_attributes = model_utils.get_custom_transforms(
    weights_name=weights_name,
    verbose=verbose,
)

train_dl, val_dl, test_dl, img_shape = get_ready_data.get_data(
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE,
    balance_data=config.BALANCE_DATA,
    custom_transforms=custom_transforms,
    train_val_test_proportions=config.TRAIN_VAL_TEST_PROPORTIONS,
    no_resizing=resize_in_attributes,
    tolerance=config.TOLERANCE,
    augmentation_proportion=config.DATA_AUGMENTATION_PROPORTION,
    random_seed=config.RANDOM_SEED,
    verbose=verbose,
)

# check image shape
print(f'img_shape: {img_shape}')

batched_img_tag = next(iter(train_dl))
batched_img_shape = batched_img_tag[0].shape
print(f'batched_img_shape: {batched_img_shape}')
print(f'batched Target shape: {batched_img_tag[1].shape}')
# remove batch dimension
img_shape = batched_img_shape[1:]
print(f'img_shape: {img_shape}')

training_history = training.train(
    model=model,
    training_data=train_dl,
    validation_data=val_dl,
    num_epochs=config.NUM_EPOCHS,
    learning_rate=config.LEARNING_RATE,
    loss_function_name=config.LOSS_FUNCTION_NAME,
    optimizer_name=config.OPTIMIZER_NAME,
    device=device,
    verbose=verbose,
    save_model=config.SAVE_MODEL,
    save_path=global_constants.MODEL_OUTPUT_DIR,
    metrics=config.METRICS,
    custom_transforms=custom_transforms,
)
print(f'training_history:\n{training_history}')

test_loss, metric_evaluations = evaluation.eval(
    model=model,
    test_data=test_dl,
    loss_function_name=config.LOSS_FUNCTION_NAME,
    device=device,
    display_confusion_matrix=config.DISPLAY_CONFUSION_MATRIX,
    metrics=config.METRICS,
    save_results=config.SAVE_MODEL,
    save_path=global_constants.MODEL_OUTPUT_DIR,
    verbose=verbose,
)
print(f'test_loss: {test_loss}')
print(f'test_metric_evaluations: {metric_evaluations}')
