import datetime
import torchvision.transforms.functional as tf

import utils
import config
import global_constants
from models import training, evaluation, model_utils
from data_preprocessing import get_ready_data


# for fine-tuning pretrained models, not for training new custom models
verbose = 2
device = utils.get_available_device(verbose=verbose)
num_classes = len(global_constants.TREE_INFORMATION)
print('Initial date and time:')
global_start = datetime.datetime.now()
print(global_start.strftime('%Y-%m-%d-%H:%M:%S'))

# keys here should match the names of the variables in config.py with '_list' appended
# e.g. 'augment_data_list' is the key for the list of values for the proportion of data to augment
search_space = {
    'data_augmentation_proportion_list': [1, 5, 20],
    'batch_size_list': [8, 16, 32, 64],
    'learning_rate_list': [0.001, 0.0005, 0.0001],
    'epochs_list': [5, 10, 20],
    'optimizer_name_list': ['Adam', 'RMSprop'],
    'balance_data_list': [True, False],
    'model_spec_list': [  # warning: case-sensitive names (model_name, weights_name)
        ('regnet_y_1_6gf', 'RegNet_Y_1_6GF_Weights.IMAGENET1K_V1'),
        # ('regnet_y_128gf', 'RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1'),
    ],
}
# number of tests for each configuration, to have a more accurate estimate of the performance
num_tests_for_configuration: int = 5

for model_spec in search_space['model_spec_list']:
    model_name, weights_name = model_spec

    # load model
    model, preprocess = model_utils.get_torchvision_model(
        model_name=model_name,
        weights_name=weights_name,
        training=True,
        num_classes=num_classes,
    )
    model.to(device=device)
    attributes = dir(preprocess)
    resize_in_attributes = False
    for attribute in attributes:
        if 'resize' in attribute.lower():
            resize_in_attributes = True
            break
    custom_transforms = [tf.to_tensor, preprocess]

    for batch_size in search_space['batch_size_list']:
        for balance_data in search_space['balance_data_list']:
            for data_augmentation_proportion in search_space['data_augmentation_proportion_list']:
                train_dl, val_dl, test_dl, img_shape = get_ready_data.get_data(
                    batch_size=batch_size,
                    shuffle=config.SHUFFLE,
                    balance_data=balance_data,
                    custom_transforms=custom_transforms,
                    train_val_test_proportions=config.TRAIN_VAL_TEST_PROPORTIONS,
                    no_resizing=resize_in_attributes,
                    tolerance=config.TOLERANCE,
                    augmentation_proportion=data_augmentation_proportion,
                    random_seed=config.RANDOM_SEED,
                    verbose=verbose,
                )

                for optimizer_name in search_space['optimizer_name_list']:
                    for learning_rate in search_space['learning_rate_list']:
                        for epochs in search_space['epochs_list']:
                            for i in range(num_tests_for_configuration):
                                training_history = training.train(
                                    model=model,
                                    training_data=train_dl,
                                    validation_data=val_dl,
                                    epochs=epochs,
                                    learning_rate=learning_rate,
                                    loss_function_name=config.LOSS_FUNCTION_NAME,
                                    optimizer_name=optimizer_name,
                                    device=device,
                                    verbose=verbose,
                                    save_model=False,
                                    custom_transforms=custom_transforms,
                                )

                                test_loss, metric_evaluations = evaluation.eval(
                                    model=model,
                                    test_data=test_dl,
                                    loss_function_name=config.LOSS_FUNCTION_NAME,
                                    device=device,
                                    display_confusion_matrix=False,
                                    metrics=config.METRICS,
                                    save_results=False,
                                    verbose=verbose,
                                )
