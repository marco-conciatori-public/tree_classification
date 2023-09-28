import json
import os
import numpy as np

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
    confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype=np.int64)
    for i in range(len(true_values)):
        confusion_matrix[true_values[i], predictions[i]] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(
        confusion_matrix,
        cmap=plt.cm.Blues,
        alpha=0.8,
        norm=colors.LogNorm(),
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
