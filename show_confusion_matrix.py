import json

import utils
import global_constants
from visualization import visualization_utils


def show_confusion_matrix_():
    # parameters
    model_id = int(input('Insert model id number: '))
    partial_name = str(input('Insert name or part of the name to distinguish between models with the same id number: '))

    _, meta_data_path = utils.get_path_by_id(
        partial_name=partial_name,
        model_id=model_id,
        folder_path=global_constants.MODEL_OUTPUT_DIR,
    )

    with open(meta_data_path, 'r') as json_file:
        meta_data = json.load(json_file)

    tag_list = meta_data['test_confusion_matrix']['true_values']
    prediction_list = meta_data['test_confusion_matrix']['predictions']
    print(f'len tag_list: {len(tag_list)}')
    print(f'len prediction_list: {len(prediction_list)}')
    # Plot the confusion matrix
    visualization_utils.display_cm(true_values=tag_list, predictions=prediction_list, save_img=False)


if __name__ == '__main__':
    show_confusion_matrix_()
