import torch
import warnings
import datetime
from pathlib import Path

import global_constants as gc


def get_available_id(partial_name: str, folder_path: str) -> int:
    pure_path = Path(folder_path)
    if pure_path.exists():
        matching_paths = pure_path.glob(f'{partial_name}*')
        current_ids = set()
        for path in matching_paths:
            # also remove the separator character between the model name and model id
            path_name_removed = path.name[len(partial_name) + 1:]
            counter = 0
            for character in path_name_removed:
                if character.isdigit():
                    counter += 1
                else:
                    break
            last_id = path_name_removed[:counter]
            last_id = int(last_id)
            current_ids.add(last_id)

        if len(current_ids) == 0:
            return 0
        max_id = max(current_ids)
        # if there are no holes in the serial number we want to have in the complete_set the next
        # biggest free number. This requires +2 instead of +1 because range() exclude the right boundary.
        complete_set = set(range(0, max_id + 2))
        difference = complete_set - current_ids
        min_free_id = min(difference)
        return min_free_id

    return 0


def get_available_device(verbose: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        warnings.warn('GPU not found, using CPU')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    if verbose >= 1:
        print(f'Device: {device}')

    return device


def identify_model(parameters: dict) -> (str, int):
    if 'model_partial_name' not in parameters or parameters['model_partial_name'] is None:
        model_partial_name = str(input('Insert name or part of the name of a model: '))
    else:
        model_partial_name = parameters['model_partial_name']
    if 'model_id' not in parameters or parameters['model_id'] is None:
        model_id = int(input('Insert model id number: '))
    else:
        model_id = parameters['model_id']
    return model_partial_name, model_id


def get_path_by_id(model_id: int, folder_path: str, model_partial_name: str = ''):
    pure_path = Path(folder_path)
    assert pure_path.exists(), f'the folder_path {folder_path} does not exists'

    # returns a GENERATOR that YELDS all the file paths matching the string
    # '[.-]' is necessary because otherwise for example id 1 would also match id 10, 11 etc...
    # '[.-]' matches a single character that is either a '.' or a '-'
    search_string = f'*{gc.EXTERNAL_PARAMETER_SEPARATOR}{model_id}[.-]*'
    if model_partial_name != '':
        search_string = f'*{model_partial_name}{search_string}'
    matching_paths = pure_path.glob(search_string)

    # transform the generator into a string, it is not possible to use len() on a generator
    matching_paths = list(matching_paths)
    n_matches = len(matching_paths)
    assert n_matches > 0, f'no matches found with model_partial_name "{model_partial_name}" ' \
                          f'and model_id "{model_id}" in folder_path "{folder_path}"'
    if n_matches == 1:
        warnings.warn(f'expected 2 matches with model_partial_name "{model_partial_name}" and'
                      f' model_id "{model_id}" in folder_path "{folder_path}". One for the'
                      f' model, the other for the meta data file')
    assert n_matches < 3, f'more than 2 match found with model_partial_name "{model_partial_name}"' \
                          f' and model_id "{model_id}" in folder_path "{folder_path}"'

    model_path = matching_paths[0]
    meta_data_path = matching_paths[1]
    # distinguish between model and meta data file
    if matching_paths[0].name.find(gc.INFO_FILE_NAME) != -1:
        model_path = matching_paths[1]
        meta_data_path = matching_paths[0]
    return model_path, meta_data_path


def pretty_print_dict(data, _level: int = 0):
    if type(data) == dict:
        for key in data:
            for i in range(_level):
                print('\t', end='')
            print(f'{key}:')
            pretty_print_dict(data[key], _level=_level + 1)
    else:
        for i in range(_level):
            print('\t', end='')
        print(data)


def timedelta_format(initial_time, final_time, truncate_seconds: bool = True):
    time_delta = final_time - initial_time
    if truncate_seconds:
        # remove times below seconds
        time_delta = time_delta - datetime.timedelta(microseconds=time_delta.microseconds)
    return time_delta


def to_bold_string(string: str):
    return f'\033[1m{string}\033[0m'


def get_species_id_by_name(species_name: str,
                           class_information: dict,
                           species_language: str = gc.SPECIES_LANGUAGE,
                           ) -> int:
    # returns the species_id of the species_name
    # if the species_name is not found returns -1
    for species_id in class_information:
        if class_information[species_id][species_language] == species_name:
            return species_id
    return -1
