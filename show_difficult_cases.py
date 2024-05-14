import cv2
import torch
import matplotlib.pyplot as plt

import utils
import global_constants
from import_args import args
from models import model_utils
from data_preprocessing import get_ready_data, data_loading


def show_difficult_cases_(**kwargs):
    # import parameters
    parameters = args.import_and_check(yaml_path=global_constants.CONFIG_PARAMETER_PATH, **kwargs)

    parameters['device'] = torch.device('cpu')
    parameters['shuffle'] = False
    parameters['use_targets'] = True
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

    img_list, tag_list, class_information_from_data = data_loading.load_data(
        data_path=parameters['data_path'],
        use_targets=parameters['use_targets'],
        use_only_classes=parameters['use_only_classes'],
        verbose=parameters['verbose'],
    )

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
            # top_class = prediction.argmax()

            true_class = tag_list[img_index]
            prediction_of_true_class = prediction[true_class]
            worst_predictions.append((prediction_of_true_class, img_index, prediction))

    print('Find the worst predictions...')
    # sort worst_predictions
    worst_predictions.sort(key=lambda x: x[0])

    # load images again because the transformed version is quite different from the original one.
    # The transformation is due to the "custom transforms" of torchvision models.
    # Since shuffling is disabled, the order of the reloaded images is the same
    img_list, tag_list, class_information_from_data = data_loading.load_data(
        data_path=parameters['data_path'],
        use_targets=parameters['use_targets'],
        use_only_classes=parameters['use_only_classes'],
        verbose=parameters['verbose'],
    )
    for i in range(len(worst_predictions)):
        if i >= parameters['worst_n_predictions']:
            break
        prediction_of_true_class, img_index, prediction = worst_predictions[i]

        print('-------------------')
        true_name = class_information_from_data[tag_list[img_index]][global_constants.SPECIES_LANGUAGE]
        print(f'TRUE LABEL: {true_name}')
        print('NETWORK EVALUATION:')
        for tree_class in range(len(prediction)):
            # if prediction[tree_class] >= config.TOLERANCE:
            text = f' - {class_information_from_data[tree_class][global_constants.SPECIES_LANGUAGE]}: ' \
                   f'{round(prediction[tree_class] * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))}%'
            print(text)

        # show image
        img = img_list[img_index]
        # convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # window_name = f'patch {img_index}'
        fig = plt.figure(figsize=(1, 1))
        fig.figimage(img)
        # fig.set_frameon(False)
        plt.axis('off')
        fig.tight_layout()
        # plt.imshow(img)
        fig.show()
        fig.waitforbuttonpress(-1)
        plt.close()


if __name__ == '__main__':
    show_difficult_cases_()
