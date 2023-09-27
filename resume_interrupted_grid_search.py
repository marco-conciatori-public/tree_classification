import json
import copy
import datetime
from pathlib import Path

import utils
import global_constants
from data_preprocessing import get_ready_data
from models import training, evaluation, model_utils


# for fine-tuning pretrained models, not for training new custom models
verbose = 0
interrupted = False
print('Initial date and time:')
global_start_time = datetime.datetime.now()
print(global_start_time.strftime('%Y-%m-%d-%H:%M:%S'))
resume_from_file = 'configuration_and_results_0.json'
partial_results_path = global_constants.PARAMETER_SEARCH_OUTPUT_DIR + resume_from_file
with open(partial_results_path, 'r') as json_file:
    partial_content = json.load(json_file)

assert partial_content['interrupted'], 'This script is only for resuming interrupted grid search. The file provided' \
                                       ' is not for an interrupted grid search.'

num_classes = partial_content['num_classes']
# number of tests for each configuration, to have a more accurate estimate of the performance
num_tests_for_configuration: int = partial_content['num_tests_for_configuration']
search_space = partial_content['search_space']
data_path = partial_content['data_path']
device = partial_content['device']
# compute number of combinations
num_different_configurations = 1
for value in search_space.values():
    num_different_configurations *= len(value)
print(f'num_different_configurations: {num_different_configurations}')
utils.pretty_print_dict(data=search_space)
print(f'num_tests_for_configuration: {num_tests_for_configuration}')

# find last valid configuration tested
results = partial_content['results']
last_configuration_counter = 0
for result in results:
    if result['configuration_counter'] > last_configuration_counter:
        last_configuration_counter = result['configuration_counter']
configuration_counter = 0
try:
    print(f'\nResuming grid search from configuration {last_configuration_counter}...')
    for model_spec in search_space['model_spec_list']:
        model_architecture, model_version, weights_name = model_spec
        print(f'model architecture: {model_architecture}, model version: {model_version}, with weights: {weights_name}')
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
                        data_path=data_path,
                        batch_size=batch_size,
                        shuffle=partial_content['shuffle'],
                        balance_data=balance_data,
                        custom_transforms=custom_transforms,
                        train_val_test_proportions=partial_content['train_val_test_proportions'],
                        no_resizing=resize_in_attributes,
                        augmentation_proportion=data_augmentation_proportion,
                        random_seed=partial_content['random_seed'],
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

                                    if configuration_counter <= last_configuration_counter:
                                        configuration_counter += 1
                                        continue

                                    if freeze_layers and weights_name is None:
                                        print('\t\t\t\t\t\t\t\tSkipping this configuration because incompatible'
                                              ' parameter values (freeze_layers=True and weights_name=None)')
                                        configuration_counter += 1
                                        continue

                                    start_time = datetime.datetime.now()
                                    average_loss_test = 0
                                    average_metrics_test = {metric_name: 0
                                                            for metric_name in partial_content['metrics']}
                                    pretrained_model_parameters = {
                                        'model_architecture': model_architecture,
                                        'model_version': model_version,
                                        'weights_name': weights_name,
                                        'freeze_layers': freeze_layers,
                                    }
                                    for i in range(num_tests_for_configuration):
                                        model = model_utils.get_torchvision_model(
                                            pretrained_model_parameters=pretrained_model_parameters,
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
                                            loss_function_name=partial_content['loss_function_name'],
                                            optimizer_name=optimizer_name,
                                            device=device,
                                            verbose=verbose,
                                            save_model=False,
                                            custom_transforms=custom_transforms,
                                            extra_info_to_save=None,
                                        )

                                        test_loss, metric_evaluations = evaluation.eval(
                                            model=model,
                                            test_data=test_dl,
                                            loss_function_name=partial_content['loss_function_name'],
                                            device=device,
                                            display_confusion_matrix=False,
                                            metrics=partial_content['metrics'],
                                            save_results=False,
                                            verbose=verbose,
                                        )
                                        average_loss_test += test_loss
                                        for metric_name, metric_evaluation in metric_evaluations.items():
                                            average_metrics_test[metric_name] += metric_evaluation

                                    average_loss_test = average_loss_test / num_tests_for_configuration
                                    for metric_name in partial_content['metrics']:
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
                                            'model': (model_version, weights_name),
                                            'freeze_layers': freeze_layers,
                                            'time_used': str(time_delta),
                                            'configuration_counter': configuration_counter,
                                        },
                                    )
                                    configuration_counter += 1

except Exception as e:
    print(e)
    interrupted = True
    end_time = datetime.datetime.now()
    print(f'Computation interrupted after: {utils.timedelta_format(global_start_time, end_time)}')
    print('Saving partial results...')

print('Test ended')
print(f'Results:\n{results}')
global_end_time = datetime.datetime.now()
total_duration = utils.timedelta_format(global_start_time, global_end_time)
print(f'Total duration: {total_duration}')

# numpy numbers/arrays are not json serializable
content = copy.deepcopy(partial_content)
content['total_duration'] = f'{partial_content["total_duration"]} + {total_duration}'
content['conclusion_date'] = global_end_time.strftime('%Y-%m-%d-%H:%M:%S')
content['interrupted'] = interrupted
content['results'] = results

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
