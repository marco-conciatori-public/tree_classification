import torch

import utils
import global_constants as gc
from import_args import args
from models import model_utils
from metrics import metric_utils
from visualization import visualization_utils
from data_preprocessing import data_loading, get_class


def show_difficult_cases_(**kwargs):
    # import parameters
    parameters = args.import_and_check(yaml_path=gc.CONFIG_PARAMETER_PATH, **kwargs)
    parameters['device'] = torch.device('cpu')
    parameters['shuffle'] = False
    parameters['use_targets'] = True
    model_partial_name, model_id = utils.identify_model(parameters=parameters)

    # load model
    model_path, info_path = utils.get_path_by_id(
        model_partial_name=model_partial_name,
        model_id=model_id,
        folder_path=gc.MODEL_OUTPUT_DIR,
    )
    loaded_model, custom_transforms, meta_data = model_utils.load_model(
        model_path=model_path,
        device=parameters['device'],
        training_mode=False,
        meta_data_path=info_path,
        verbose=parameters['verbose'],
    )
    class_information_from_model = meta_data['class_information']

    # load data
    img_list, tag_list, class_information_from_data = data_loading.load_data(
        data_path=parameters['data_path'],
        use_targets=parameters['use_targets'],
        model_class_information=class_information_from_model,
        use_only_classes=parameters['use_only_classes'],
        verbose=parameters['verbose'],
    )
    if 'launched_from_notebook' in kwargs and kwargs['launched_from_notebook']:
        if parameters['verbose'] >= 2:
            print('Execution in notebook mode. Returning output instead of printing.')

    # use model
    softmax = torch.nn.Softmax(dim=0)
    worst_predictions = []
    with torch.set_grad_enabled(False):
        for img_index in range(len(img_list)):
            img = img_list[img_index]

            # apply custom transforms
            if custom_transforms is not None:
                for transform in custom_transforms:
                    img = transform(img)

            # add batch dimension
            img = img.unsqueeze(0)
            prediction = loaded_model(img)
            prediction = prediction.squeeze(0)
            prediction = softmax(prediction)
            prediction = prediction.numpy()

            true_class_id_from_data = tag_list[img_index]
            true_class_id_from_model = get_class.convert_class_id_from_different_sets(
                class_id=true_class_id_from_data,
                class_information_source=class_information_from_data,
                class_information_destination=class_information_from_model,
            )
            prediction_of_true_class = 0
            if true_class_id_from_model != -1:
                prediction_of_true_class = prediction[true_class_id_from_model]
            worst_predictions.append((prediction_of_true_class, img_index, prediction))

    print('Finding the worst predictions...')
    # sort worst_predictions
    worst_predictions.sort(key=lambda x: x[0])

    # load images again because the transformed version is quite different from the original one.
    # The transformation is due to the "custom transforms" of torchvision models.
    # Since shuffling is disabled, the order of the reloaded images is the same
    img_list, tag_list, class_information_from_data = data_loading.load_data(
        data_path=parameters['data_path'],
        use_targets=parameters['use_targets'],
        model_class_information=class_information_from_model,
        use_only_classes=parameters['use_only_classes'],
        verbose=0,
    )
    notebook_output = []
    for i in range(len(worst_predictions)):
        if i >= parameters['worst_n_predictions']:
            break
        _, img_index, prediction = worst_predictions[i]
        true_name = class_information_from_data[tag_list[img_index]][gc.SPECIES_LANGUAGE]

        if 'launched_from_notebook' in kwargs and kwargs['launched_from_notebook']:
            element_info = {}
            element_info['true_label'] = true_name
            element_info['prediction'] = {}
            for tree_class_local_index in range(len(prediction)):
                # if prediction[tree_class_local_index] >= config.TOLERANCE:
                element_info['prediction'][class_information_from_model[tree_class_local_index][gc.SPECIES_LANGUAGE]
                ] = metric_utils.format_value(value=prediction[tree_class_local_index], as_percentage=True)
            element_info['img'] = img_list[img_index]
            element_info['tag'] = tag_list[img_index]
            notebook_output.append(element_info)

        else:
            print('-------------------')
            print(f'TRUE LABEL: {true_name}')
            print('NETWORK EVALUATION:')
            for tree_class_local_index in range(len(prediction)):
                # if prediction[tree_class_local_index] >= config.TOLERANCE:
                text = f' - {class_information_from_model[tree_class_local_index][gc.SPECIES_LANGUAGE]}: ' \
                       f'{metric_utils.format_value(value=prediction[tree_class_local_index], as_percentage=True)}'
                print(text)

            # show image
            visualization_utils.display_img(img=img_list[img_index])

    if 'launched_from_notebook' in kwargs and kwargs['launched_from_notebook']:
        return notebook_output


if __name__ == '__main__':
    show_difficult_cases_()
