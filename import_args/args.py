import os

import utils
import global_constants
from import_args import from_yaml, from_command_line, from_function_arguments


def import_and_check(yaml_path, **kwargs) -> dict:
    max_iterations = 5
    level_up = 0
    data_dict = None
    while data_dict is None:
        try:
            data_dict = from_yaml.read_config(yaml_path)
        except FileNotFoundError:
            if level_up == max_iterations:
                raise
            level_up += 1
            os.chdir("..")

    # command line arguments have priority over yaml arguments
    data_dict = from_command_line.update_config(data_dict)
    # function arguments have priority over yaml and command line arguments
    data_dict = from_function_arguments.update_config(data_dict, **kwargs)

    # check that proportions adds up to 1, except for rounding errors
    tolerance = data_dict['tolerance']
    assert 1 - tolerance < sum(data_dict['train_val_test_proportions']) < 1 + tolerance, \
        f'The values of train_val_test_proportions must add up to 1 +/- {tolerance}'

    device = utils.get_available_device(verbose=data_dict['verbose'])
    data_dict['device'] = device
    if 'data_path' not in data_dict or data_dict['data_path'] is None:
        data_dict['data_path'] = global_constants.DATA_PATH + data_dict['input_data_folder']

    return data_dict
