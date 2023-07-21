import json
import os

import global_constants


def load_evaluation(file_number: int) -> dict:
    max_iterations = 5
    level_up = 0
    hp_evaluation = None
    file_path = f'{global_constants.PARAMETER_SEARCH_OUTPUT_DIR}' \
                f'{global_constants.PARAMETER_SEARCH_FILE_NAME}_{file_number}.json'
    while hp_evaluation is None:
        try:
            with open(file_path, 'r') as json_file:
                hp_evaluation = json.load(json_file)
        except FileNotFoundError:
            if level_up == max_iterations:
                raise
            level_up += 1
            os.chdir("..")

    return hp_evaluation


def identify_tested_hp(search_space: dict, excluded_key_list: list = None) -> list:
    hp_to_plot = []
    for key in search_space:
        if (excluded_key_list is None) or (key not in excluded_key_list):
            value = search_space[key]
            if isinstance(value, list):
                if len(value) > 1:
                    hp_to_plot.append(key)

    return hp_to_plot


def extract_parameter_keys(parameters_to_plot: list) -> list:
    parameter_keys = []
    for parameter in parameters_to_plot:
        parameter_without_suffix = parameter[:-5]
        if parameter_without_suffix == 'model_spec':
            parameter_without_suffix = 'model'
        parameter_keys.append(parameter_without_suffix)
    return parameter_keys
