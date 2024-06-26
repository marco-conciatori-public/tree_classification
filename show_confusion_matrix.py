import json

import utils
import global_constants as gc
from visualization import visualization_utils


def show_confusion_matrix_():
    # parameters
    model_partial_name, model_id = utils.identify_model(parameters={})
    _, meta_data_path = utils.get_path_by_id(
        model_partial_name=model_partial_name,
        model_id=model_id,
        folder_path=gc.MODEL_OUTPUT_DIR,
    )

    with open(meta_data_path, 'r') as json_file:
        meta_data = json.load(json_file)

    tag_list = meta_data['test_confusion_matrix']['true_values']
    prediction_list = meta_data['test_confusion_matrix']['predictions']
    print(f'len tag_list: {len(tag_list)}')
    print(f'len prediction_list: {len(prediction_list)}')

    # Plot the confusion matrix
    visualization_utils.display_cm(
        true_values=tag_list,
        predictions=prediction_list,
        class_information=meta_data['class_information'],
        save_img=False,
    )


if __name__ == '__main__':
    show_confusion_matrix_()
