import torch
import warnings
import datetime
import numpy as np
from pathlib import Path

import global_constants


def get_available_id(partial_name: str, folder_path: str) -> int:
    pure_path = Path(folder_path)
    if pure_path.exists():
        matching_paths = pure_path.glob(f'{partial_name}*')
        current_ids = set()
        for path in matching_paths:
            # also remove the separator character between the model name and model id
            path_name_removed = path.name[len(partial_name) + 1:]
            last_id = ''
            counter = 0
            while path_name_removed[counter].isdigit():
                counter += 1
            last_id += path_name_removed[:counter]
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


def check_split_proportions(train_val_test_proportions: list, tolerance: float):
    # check that proportions adds up to 1, except for rounding errors
    assert 1 - tolerance < sum(train_val_test_proportions) < 1 + tolerance, \
        f'The values of train_val_test_proportions must add up to 1 +/- {tolerance}' \
        f' They add up to {sum(train_val_test_proportions)}'


def get_path_by_id(model_id: int, folder_path: str, partial_name: str = ''):
    pure_path = Path(folder_path)
    assert pure_path.exists(), f'ERROR: The folder_path {folder_path} does not exists'

    # returns a GENERATOR that YELDS all the file paths matching the string
    search_string = f'*{global_constants.EXTERNAL_PARAMETER_SEPARATOR}{model_id}*'
    if partial_name != '':
        search_string = f'*{partial_name}{search_string}'
    matching_paths = pure_path.glob(search_string)

    # transform the generator into a string, it is not possible to use len() on a generator
    matching_paths = list(matching_paths)
    n_matches = len(matching_paths)
    assert n_matches > 0, f'ERROR: No matches found with partial_name "{partial_name}" ' \
                          f'and model_id "{model_id}" in folder_path "{folder_path}"'
    if n_matches == 1:
        warnings.warn(f'expected 2 matches with partial name "{partial_name}" and'
                      f' model_id "{model_id}" in folder_path "{folder_path}". One for the'
                      f' model, the other for the meta data file')
    assert n_matches < 3, f'ERROR: More than 2 match found with partial_name "{partial_name}"' \
                          f' and model_id "{model_id}" in folder_path "{folder_path}"'

    model_path = matching_paths[0]
    meta_data_path = matching_paths[1]
    if matching_paths[0].name.find(global_constants.INFO_FILE_NAME) != -1:
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


def display_cm(true_values, predictions, labels=None):
    # Plot the confusion matrix
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    if labels is None:
        labels = []
        for el in global_constants.TREE_INFORMATION.values():
            labels.append(el[global_constants.TREE_NAME_TO_SHOW])

    num_classes = len(global_constants.TREE_INFORMATION)
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(true_values)):
        confusion_matrix[true_values[i], predictions[i]] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(
        confusion_matrix,
        cmap=plt.cm.Blues,
        alpha=0.8,
        norm=colors.LogNorm(vmin=confusion_matrix.min(), vmax=confusion_matrix.max()),
    )
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j, y=i, s=int(confusion_matrix[i, j]), va='center', ha='center', size=10)

    ax.xaxis.set_ticks_position("bottom")
    plt.xticks(range(num_classes), labels, rotation=60, fontsize=10)
    plt.yticks(range(num_classes), labels, fontsize=10)
    plt.xlabel('Predictions', fontsize=17)
    plt.ylabel('True values', fontsize=17)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()


def timedelta_format(initial_time, final_time, truncate_seconds: bool = True):
    time_delta = final_time - initial_time
    if truncate_seconds:
        # remove times below seconds
        time_delta = time_delta - datetime.timedelta(microseconds=time_delta.microseconds)
    return time_delta


def to_bold_string(string: str):
    return f'\033[1m{string}\033[0m'


def get_tree_name(species_id: int, name_type: str = global_constants.TREE_NAME_TO_SHOW):
    return global_constants.TREE_INFORMATION[species_id][name_type]
