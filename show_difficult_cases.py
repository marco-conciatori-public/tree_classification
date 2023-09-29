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
    model_id = int(input('Insert model id number: '))
    partial_name = str(input('Insert name or part of the name to distinguish between models with the same id number: '))

    model_path, info_path = utils.get_path_by_id(
        partial_name=partial_name,
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

    train_dl, val_dl, test_dl, img_shape = get_ready_data.get_data(
        data_path=parameters['data_path'],
        shuffle=False,
        batch_size=1,
        balance_data=False,
        train_val_test_proportions=parameters['train_val_test_proportions'],
        # standard_img_dim=config.IMG_DIM,
        custom_transforms=custom_transforms,
        augmentation_proportion=1,
        random_seed=parameters['random_seed'],
        verbose=parameters['verbose'],
    )

    print('Get all the predictions...')
    img_list = []
    tag_list = []
    for batch in train_dl:
        observation_batch, target_batch = batch
        img_list.append(observation_batch)
        tag_list.append(target_batch.squeeze(0).item())
    for batch in val_dl:
        observation_batch, target_batch = batch
        img_list.append(observation_batch)
        tag_list.append(target_batch.squeeze(0).item())
    for batch in test_dl:
        observation_batch, target_batch = batch
        img_list.append(observation_batch)
        tag_list.append(target_batch.squeeze(0).item())

    softmax = torch.nn.Softmax(dim=0)
    worst_predictions = []
    with torch.set_grad_enabled(False):
        for img_index in range(len(img_list)):
            img = img_list[img_index]

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
    img_list, tag_list = data_loading.load_data(
        data_path=parameters['data_path'],
        verbose=0,
    )
    for i in range(parameters['worst_n_predictions']):
        if i > parameters['worst_n_predictions']:
            break
        prediction_of_true_class, img_index, prediction = worst_predictions[i]

        print('-------------------')
        print(f'TRUE LABEL: '
              f'{utils.get_tree_name(tag_list[img_index]).upper()}')
        print('NETWORK EVALUATION:')
        for tree_class in range(len(prediction)):
            if prediction[tree_class] >= parameters['tolerance']:
                print(f' - {utils.get_tree_name(tree_class)}: '
                      f'{round(prediction[tree_class] * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))}')

        # show image
        img = img_list[img_index]
        # convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        plt.close()
        # cv2.imshow(
        #     winname=utils.get_tree_name(tag_list[img_index]).upper(),
        #     mat=img,
        # )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    show_difficult_cases_()
