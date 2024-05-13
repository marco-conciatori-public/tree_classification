import json
# import torch
# import torchmetrics

import utils
import global_constants
from visualization import visualization_utils


def show_confusion_matrix_():
    # parameters
    model_partial_name, model_id = utils.identify_model(parameters={})
    _, meta_data_path = utils.get_path_by_id(
        model_partial_name=model_partial_name,
        model_id=model_id,
        folder_path=global_constants.MODEL_OUTPUT_DIR,
    )

    with open(meta_data_path, 'r') as json_file:
        meta_data = json.load(json_file)

    tag_list = meta_data['test_confusion_matrix']['true_values']
    prediction_list = meta_data['test_confusion_matrix']['predictions']
    print(f'len tag_list: {len(tag_list)}')
    print(f'len prediction_list: {len(prediction_list)}')

    # accuracy_none = torchmetrics.Accuracy(task='multiclass', average='none', num_classes=5)
    # accuracy_micro = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=5)
    # accuracy_macro = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=5)
    # recall_none = torchmetrics.Recall(task='multiclass', average='none', num_classes=5)
    # recall_micro = torchmetrics.Recall(task='multiclass', average='micro', num_classes=5)
    # recall_macro = torchmetrics.Recall(task='multiclass', average='macro', num_classes=5)
    # accuracy_none.update(torch.tensor(prediction_list), torch.tensor(tag_list))
    # accuracy_micro.update(torch.tensor(prediction_list), torch.tensor(tag_list))
    # accuracy_macro.update(torch.tensor(prediction_list), torch.tensor(tag_list))
    # recall_none.update(torch.tensor(prediction_list), torch.tensor(tag_list))
    # recall_micro.update(torch.tensor(prediction_list), torch.tensor(tag_list))
    # recall_macro.update(torch.tensor(prediction_list), torch.tensor(tag_list))
    # print(f'accuracy_micro: {accuracy_micro.compute()}')
    # print(f'accuracy_macro: {accuracy_macro.compute()}')
    # print(f'accuracy_none: {accuracy_none.compute()}')
    # print(f'recall_micro: {recall_micro.compute()}')
    # print(f'recall_macro: {recall_macro.compute()}')
    # print(f'recall_none: {recall_none.compute()}')

    # Plot the confusion matrix
    visualization_utils.display_cm(
        true_values=tag_list,
        predictions=prediction_list,
        class_information=meta_data['class_information'],
        save_img=False,
    )


if __name__ == '__main__':
    show_confusion_matrix_()
