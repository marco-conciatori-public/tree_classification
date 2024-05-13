import cv2
import torch

import utils
import global_constants
from import_args import args
from models import model_utils
from data_preprocessing import data_loading


def use_model_(**kwargs):
    # import parameters
    parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH, **kwargs)
    use_targets = parameters['use_targets']

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
    class_information = meta_data['class_information']

    img_list, tag_list, class_information_from_data = data_loading.load_data(
        data_path=parameters['data_path'],
        use_targets=use_targets,
        use_only_classes=parameters['use_only_classes'],
        verbose=parameters['verbose'],
    )
    if parameters['jump'] > 1:
        img_list = img_list[::parameters['jump']]
        print(f'img_list length after "jump" selection: {len(img_list)}')
    if use_targets:
        tag_list = tag_list[::parameters['jump']]

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

            print('-------------------')
            if use_targets:
                true_name = class_information[tag_list[img_index]][global_constants.SPECIES_LANGUAGE]
                print(f'TRUE LABEL: {true_name}')
            print('NETWORK EVALUATION:')
            for tree_class in range(len(prediction)):
                # if prediction[tree_class] >= config.TOLERANCE:
                text = f' - {class_information[tree_class][global_constants.SPECIES_LANGUAGE]}: ' \
                       f'{round(prediction[tree_class] * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))}%'
                if tree_class == top_class:
                    text = utils.to_bold_string(text)
                print(text)

            img = img_list[img_index]
            window_name = f'patch {img_index}'
            if use_targets:
                window_name = true_name
            # show image
            cv2.imshow(winname=window_name, mat=img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    use_model_()
