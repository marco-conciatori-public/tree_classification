import json
import datetime
from pathlib import Path

import utils
import config
import global_constants
from data_preprocessing import get_ready_data
from models import training, evaluation, model_utils


# for fine-tuning pretrained models, not for training new custom models
verbose = 0
device = utils.get_available_device(verbose=verbose)
num_classes = len(global_constants.TREE_INFORMATION)
interrupted = False
print('Initial date and time:')
global_start_time = datetime.datetime.now()
print(global_start_time.strftime('%Y-%m-%d-%H:%M:%S'))

# number of tests for each configuration, to have a more accurate estimate of the performance
num_tests_for_configuration: int = 3
# keys here should match the names of the variables in config.py with '_list' appended
# e.g. 'augment_data_list' is the key for the list of values for the proportion of data to augment.
# each combination of values will be tested num_tests_for_configuration times
# and the average performance will be used to select the best configuration.
# use lists of one element to test only one value for that variable.
search_space = {
    'data_augmentation_proportion_list': [1, 5],
    # 'data_augmentation_proportion_list': [1],
    # 'batch_size_list': [8, 16, 32],
    'batch_size_list': [16],
    'learning_rate_list': [0.005, 0.0001],
    # 'learning_rate_list': [0.001],
    'num_epochs_list': [4, 10],
    # 'num_epochs_list': [3],
    # 'optimizer_name_list': ['Adam', 'RMSprop'],
    'optimizer_name_list': ['RMSprop'],
    'balance_data_list': [True, False],
    # 'balance_data_list': [False],
    'model_spec_list': [  # warning: case-sensitive names (model_name, weights_name)
        ('regnet_y_1_6gf', 'RegNet_Y_1_6GF_Weights.DEFAULT'),
        ('regnet_y_1_6gf', None),
        # ('regnet_y_128gf', 'RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1'),
    ],
    # to train a model from scratch, set weights_name to None
    'freeze_layers_list': [True, False],
    # 'freeze_layers_list': [True],
}
# compute number of combinations
num_different_configurations = 1
for value in search_space.values():
    num_different_configurations *= len(value)
print(f'num_different_configurations: {num_different_configurations}')
utils.pretty_print_dict(data=search_space)
print(f'num_tests_for_configuration: {num_tests_for_configuration}')

configuration_counter = 0
results = []
try:
    print('\nStarting grid search...')
    for model_spec in search_space['model_spec_list']:
        model_name, weights_name = model_spec
        print(f'model: {model_name}, with weights: {weights_name}')
        custom_transforms, resize_in_attributes = model_utils.get_custom_transforms(
            weights_name=weights_name,
            verbose=verbose,
        )

        for batch_size in search_space['batch_size_list']:
            print(f'\tbatch_size: {batch_size}')
            for balance_data in search_space['balance_data_list']:
                print(f'\t\tbalance_data: {balance_data}')
                for data_augmentation_proportion in search_space['data_augmentation_proportion_list']:
                    print(f'\t\t\tdata_augmentation_proportion: {data_augmentation_proportion}')
                    start_time = datetime.datetime.now()
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
                    end_time = datetime.datetime.now()
                    print(f'\t\t\tdata processing/loading time: {utils.timedelta_format(start_time, end_time)}')

                    for optimizer_name in search_space['optimizer_name_list']:
                        print(f'\t\t\t\toptimizer: {optimizer_name}')
                        for learning_rate in search_space['learning_rate_list']:
                            print(f'\t\t\t\t\tlearning_rate: {learning_rate}')
                            for num_epochs in search_space['num_epochs_list']:
                                print(f'\t\t\t\t\t\tnum_epochs: {num_epochs}')
                                for freeze_layers in search_space['freeze_layers_list']:
                                    print(f'\t\t\t\t\t\t\tfreeze_layers: {freeze_layers}')
                                    print(f'\t\t\t\t\t\t\t\tconfiguration {configuration_counter + 1} / '
                                          f'{num_different_configurations}')

                                    if freeze_layers and weights_name is None:
                                        print('\t\t\t\t\t\t\t\tSkipping this configuration because incompatible'
                                              ' parameter values (freeze_layers=True and weights_name=None)')
                                        configuration_counter += 1
                                        continue

                                    start_time = datetime.datetime.now()
                                    average_loss_test = 0
                                    average_metrics_test = {metric_name: 0 for metric_name in config.METRICS}
                                    for i in range(num_tests_for_configuration):
                                        model = model_utils.get_torchvision_model(
                                            model_name=model_name,
                                            freeze_layers=freeze_layers,
                                            weights_name=weights_name,
                                            device=device,
                                            training=True,
                                            num_classes=num_classes,
                                            verbose=verbose,
                                        )

                                        training_history = training.train(
                                            model=model,
                                            training_data=train_dl,
                                            validation_data=val_dl,
                                            num_epochs=num_epochs,
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
                                        average_loss_test += test_loss
                                        for metric_name, metric_evaluation in metric_evaluations.items():
                                            average_metrics_test[metric_name] += metric_evaluation

                                    average_loss_test = average_loss_test / num_tests_for_configuration
                                    for metric_name in config.METRICS:
                                        average_metrics_test[metric_name] = average_metrics_test[metric_name] / \
                                                                                num_tests_for_configuration

                                    print(f'\t\t\t\t\t\t\t\taverage_loss_test: {average_loss_test}')
                                    end_time = datetime.datetime.now()
                                    time_delta = utils.timedelta_format(start_time, end_time)
                                    print(f'\t\t\t\t\t\t\t\t{num_tests_for_configuration} identical models trained'
                                          f' and tested in: {time_delta}')
                                    results.append(
                                        {
                                            'test_loss': average_loss_test,
                                            'test_metrics': average_metrics_test,
                                            'learning_rate': learning_rate,
                                            'balance_data': balance_data,
                                            'batch_size': batch_size,
                                            'data_augmentation_proportion': data_augmentation_proportion,
                                            'optimizer_name': optimizer_name,
                                            'num_epochs': num_epochs,
                                            'model': (model_name, weights_name),
                                            'freeze_layers': freeze_layers,
                                            'time_used': str(time_delta),
                                            'configuration_counter': configuration_counter,
                                        },
                                    )
                                    configuration_counter += 1

except KeyboardInterrupt as e:
    print(e)
    interrupted = True
    end_time = datetime.datetime.now()
    print(f'Computation interrupted after: {utils.timedelta_format(global_start_time, end_time)}')
    print('Saving partial results...')

print('Test ended')
print('Results:')
utils.pretty_print_dict(results)
global_end_time = datetime.datetime.now()
total_duration = utils.timedelta_format(global_start_time, global_end_time)
print(f'Total duration: {total_duration}')

# numpy numbers/arrays are not json serializable
content = {
    'total_duration': str(total_duration),
    'conclusion_date': global_end_time.strftime('%Y-%m-%d-%H:%M:%S'),
    'shuffle': config.SHUFFLE,
    'num_classes': num_classes,
    'num_tests_for_configuration': num_tests_for_configuration,
    'interrupted': interrupted,
    'loss_function_name': config.LOSS_FUNCTION_NAME,
    'train_val_test_proportions': config.TRAIN_VAL_TEST_PROPORTIONS,
    'tolerance': config.TOLERANCE,
    'random_seed': config.RANDOM_SEED,
    'metrics': config.METRICS,
    'search_space': search_space,
    'results': results,
}

# save results
Path(global_constants.PARAMETER_SEARCH_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
# find first free index
i = 0
while Path(f'{global_constants.PARAMETER_SEARCH_OUTPUT_DIR}{global_constants.PARAMETER_SEARCH_FILE_NAME}'
           f'{global_constants.INTERNAL_PARAMETER_SEPARATOR}{i}.json').exists():
    i += 1
with open(f'{global_constants.PARAMETER_SEARCH_OUTPUT_DIR}{global_constants.PARAMETER_SEARCH_FILE_NAME}'
          f'{global_constants.INTERNAL_PARAMETER_SEPARATOR}{i}.json', 'w') as json_file:
    json.dump(content, json_file)

print('Final date and time:')
print(global_end_time.strftime('%Y-%m-%d-%H:%M:%S'))