import global_constants
from import_args import args
from models import training, evaluation, model_utils
from data_preprocessing import get_ready_data


# import parameters
parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH)
num_classes = len(global_constants.TREE_INFORMATION)

# load model
model = model_utils.get_torchvision_model(
    pretrained_model_parameters=parameters['pretrained_model_parameters'],
    device=parameters['device'],
    training=True,
    num_classes=num_classes,
    verbose=parameters['verbose'],
)
# print(f'model:\n{model}')
custom_transforms, resize_in_attributes = model_utils.get_custom_transforms(
    weights_name=parameters['pretrained_model_parameters']['weights_name'],
    verbose=parameters['verbose'],
)

train_dl, val_dl, test_dl, img_shape = get_ready_data.get_data(
    data_path=parameters['data_path'],
    batch_size=parameters['batch_size'],
    shuffle=parameters['shuffle'],
    balance_data=parameters['balance_data'],
    custom_transforms=custom_transforms,
    train_val_test_proportions=parameters['train_val_test_proportions'],
    no_resizing=resize_in_attributes,
    augmentation_proportion=parameters['data_augmentation_proportion'],
    random_seed=parameters['random_seed'],
    verbose=parameters['verbose'],
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

parameters_to_save = {}
parameters_to_save['shuffle'] = parameters['shuffle']
parameters_to_save['random_seed'] = parameters['random_seed']
parameters_to_save['augmentation_proportion'] = parameters['data_augmentation_proportion']
parameters_to_save['balance_classes'] = parameters['balance_data']
training_history = training.train(
    model=model,
    training_data=train_dl,
    validation_data=val_dl,
    num_epochs=parameters['num_epochs'],
    learning_rate=parameters['learning_rate'],
    loss_function_name=parameters['loss_function_name'],
    optimizer_name=parameters['optimizer_name'],
    device=parameters['device'],
    verbose=parameters['verbose'],
    save_model=parameters['save_model'],
    save_path=global_constants.MODEL_OUTPUT_DIR,
    metrics=parameters['metrics'],
    custom_transforms=custom_transforms,
    extra_info_to_save=parameters_to_save,
)
print(f'training_history:\n{training_history}')

test_loss, metric_evaluations = evaluation.eval(
    model=model,
    test_data=test_dl,
    loss_function_name=parameters['loss_function_name'],
    device=parameters['device'],
    display_confusion_matrix=parameters['display_confusion_matrix'],
    metrics=parameters['metrics'],
    save_results=parameters['save_model'],
    save_path=global_constants.MODEL_OUTPUT_DIR,
    notebook_mode=parameters['notebook_mode'],
    verbose=parameters['verbose'],
)
print(f'test_loss: {test_loss}')
print(f'test_metric_evaluations: {metric_evaluations}')
