import torch

import utils
import global_constants
from import_args import args
from models import model_utils
from metrics import metric_utils
from data_preprocessing import data_loading
from visualization import visualization_utils


def use_model_(**kwargs):
    # import parameters
    parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH, **kwargs)
    use_targets = parameters['use_targets']

    # load model
    model_partial_name, model_id = utils.identify_model(parameters=parameters)
    model_path, info_path = utils.get_path_by_id(
        model_partial_name=model_partial_name,
        model_id=model_id,
        folder_path=global_constants.MODEL_OUTPUT_DIR,
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
        use_targets=use_targets,
        model_class_information=class_information_from_model,
        use_only_classes=parameters['use_only_classes'],
        verbose=parameters['verbose'],
    )
    if parameters['jump'] > 1:
        img_list = img_list[::parameters['jump']]
        print(f'img_list length after "jump" selection: {len(img_list)}')
    if use_targets:
        tag_list = tag_list[::parameters['jump']]
    if 'launched_from_notebook' in kwargs and kwargs['launched_from_notebook']:
        if parameters['verbose'] >= 2:
            print('Execution in notebook mode. Returning output instead of printing.')

    # use model
    notebook_output = []
    softmax = torch.nn.Softmax(dim=0)
    with torch.set_grad_enabled(False):
        for img_index in range(len(img_list)):
            img = img_list[img_index]

            # apply custom transforms
            if custom_transforms is not None:
                for transform in custom_transforms:
                    img = transform(img)

            if img.device != parameters['device']:
                img = img.to(parameters['device'])

            # add batch dimension
            img = img.unsqueeze(0)
            prediction = loaded_model(img)
            prediction = prediction.squeeze(0)
            prediction = softmax(prediction)
            prediction = prediction.detach().cpu().numpy()
            top_class = prediction.argmax()

            if 'launched_from_notebook' in kwargs and kwargs['launched_from_notebook']:
                element_info = {}
                print('-------------------')
                if use_targets:
                    print(f'img_index: {img_index}')
                    print(f'tag: {tag_list[img_index]}')
                    true_name = class_information_from_data[tag_list[img_index]][global_constants.SPECIES_LANGUAGE]
                    element_info['true_label'] = true_name
                    print(f'TRUE LABEL: {true_name}')
                element_info['prediction'] = {}
                for tree_class_local_index in range(len(prediction)):
                    print(f'tree_class_local_index: {tree_class_local_index}')
                    print(f'tree class name: {class_information_from_model[tree_class_local_index][global_constants.SPECIES_LANGUAGE]}')
                    # if prediction[tree_class_local_index] >= config.TOLERANCE:
                    element_info['prediction'][class_information_from_model[tree_class_local_index]
                    [global_constants.SPECIES_LANGUAGE]] = metric_utils.format_value(
                        value=prediction[tree_class_local_index],
                        as_percentage=True,
                    )
                element_info['img'] = img_list[img_index]
                element_info['tag'] = tag_list[img_index]
                notebook_output.append(element_info)

            else:
                print('-------------------')
                if use_targets:
                    true_name = class_information_from_data[tag_list[img_index]][global_constants.SPECIES_LANGUAGE]
                    print(f'TRUE LABEL: {true_name}')
                print('NETWORK EVALUATION:')
                for tree_class_local_index in range(len(prediction)):
                    # if prediction[tree_class_local_index] >= config.TOLERANCE:
                    text = f' - {class_information_from_data[tree_class_local_index][global_constants.SPECIES_LANGUAGE]}: ' \
                           f'{metric_utils.format_value(value=prediction[tree_class_local_index], as_percentage=True)}'
                    if tree_class_local_index == top_class:
                        text = utils.to_bold_string(text)
                    print(text)

                # show image
                visualization_utils.display_img(img=img_list[img_index])
    exit()
    if 'launched_from_notebook' in kwargs and kwargs['launched_from_notebook']:
        return notebook_output


if __name__ == '__main__':
    use_model_()
