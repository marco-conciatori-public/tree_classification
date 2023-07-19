import json

import utils
import global_constants


# PARAMETERS
model_id = 1
partial_name = 'regnet_y'

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
utils.display_cm(true_values=tag_list, predictions=prediction_list)
