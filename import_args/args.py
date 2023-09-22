import os

import utils
from import_args import from_yaml, from_command_line


def import_and_check(yaml_path) -> dict:
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

    data_dict = from_command_line.update_config(data_dict)

    # check that proportions adds up to 1, except for rounding errors
    tolerance = data_dict['tolerance']
    assert 1 - tolerance < sum(data_dict['train_val_test_proportions']) < 1 + tolerance, \
        f'The values of train_val_test_proportions must add up to 1 +/- {tolerance}'

    device = utils.get_available_device(verbose=data_dict['verbose'])
    data_dict['device'] = device

    return data_dict
