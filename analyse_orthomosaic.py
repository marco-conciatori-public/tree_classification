import torch
import datetime
import warnings
import numpy as np
import tifffile as tifi
import torchvision.transforms.functional as tf

import utils
import global_constants
from models import model_utils
from image_processing import orthomosaic_utils


def analyse_orthomosaic_(**kwargs):
    global_start_time = datetime.datetime.now()
    start_time = global_start_time
    # compute species probability distribution for each pixel in the image_processing image

    # in pixels
    # set patch_size to None to use the crop_size from the model. Only works for torchvision pretrained models
    # TODO: chiedere a quanti pixel corrisponde una patch reale di 2m x 2m
    # TODO: use probabilities instead of the top class
    # TODO: try with smaller stride
    # TODO: try with smaller patch_size, the dimension of the patch used for training
    # TODO: try with moving window and record only when certainty of prediction is above a certain
    #  threshold (very high, es. 96%)
    patch_size = 95
    stride = 50
    # confidence prediction probability above which the prediction is considered valid
    confidence_threshold = 0.5
    img_name_no_extension = kwargs['img_name'].split('.')[0]
    print(f'Orthomosaic used: {img_name_no_extension}')

    # load model
    device = utils.get_available_device(verbose=kwargs['verbose'])
    model_path, info_path = utils.get_path_by_id(
        partial_name=kwargs['partial_name'],
        model_id=kwargs['model_id'],
        folder_path=global_constants.MODEL_OUTPUT_DIR,
    )
    loaded_model, custom_transforms, meta_data = model_utils.load_model(
        model_path=model_path,
        device=device,
        training_mode=False,
        meta_data_path=info_path,
        verbose=kwargs['verbose'],
    )
    if custom_transforms is not None:
        preprocess = custom_transforms[1]
        if patch_size is None:
            patch_size = preprocess.crop_size[0]
        # print(f'patch_size from preprocess: {patch_size}')
    else:
        assert patch_size is not None, 'patch_size must be specified if no custom_transforms are used'
        preprocess = None
    print(f'patch_size: {patch_size}')
    end_time = datetime.datetime.now()
    print(f'load model time: {utils.timedelta_format(start_time, end_time)}')

    # load orthomosaic image
    start_time = end_time
    orthomosaic_path = global_constants.ORTHOMOSAIC_DATA_PATH + kwargs['img_name']
    orthomosaic = tifi.imread(orthomosaic_path)
    # TODO: remove this line, only for speeding up testing
    orthomosaic = orthomosaic[10000 : 11000, 10000 : 11000, :]
    # orthomosaic = orthomosaic[10000 : 15000, 15000 : 25000, :]
    print(f'orthomosaic.shape: {orthomosaic.shape}')
    # print(f'orthomosaic type: {type(orthomosaic)}')
    # print(f'orthomosaic[0, 0, 0] type: {type(orthomosaic[0, 0, 0])}')
    total_width = orthomosaic.shape[0]
    total_height = orthomosaic.shape[1]
    max_x = total_width - patch_size
    max_y = total_height - patch_size
    print(f'max_x: {max_x}, max_y: {max_y}')

    unknown_class_id = meta_data['num_classes']
    num_classes_plus_unknown = meta_data['num_classes'] + 1
    print(f'num_classes: {meta_data["num_classes"]}, num_classes_plus_unknown: {num_classes_plus_unknown}')

    # initialize species distribution
    # the last channel is used to count the number of times a pixel has been predicted
    # the second to last channel is used to count the number of times a pixel has been predicted as unknown
    species_distribution = np.zeros((total_width, total_height, num_classes_plus_unknown + 1), dtype=np.int8)

    # remove fourth channel
    orthomosaic = orthomosaic[ : , : , 0 : 3]
    print(f'remove fourth/alpha channel orthomosaic.shape: {orthomosaic.shape}')

    # TODO: check if the model is trained with images with inverted colors (because this is with normal colors)

    # to tensor
    orthomosaic = tf.to_tensor(orthomosaic)
    print(f'to tensor orthomosaic.shape: {orthomosaic.shape}')

    # # to device
    # orthomosaic = orthomosaic.to(device)
    # print(f'to device {device}, orthomosaic type: {type(orthomosaic)}')
    end_time = datetime.datetime.now()
    print(f'load orthomosaic time: {utils.timedelta_format(start_time, end_time)}')

    # loop through the orthomosaic image_processing image
    start_time = end_time
    softmax = torch.nn.Softmax(dim=0)
    # to limit the amount of ram used, one patch at a time is extracted from the orthomosaic image and fed to the model
    for x in range(0, max_x, stride):
        for y in range(0, max_y, stride):
            # print(f'x: {x}, y: {y}')
            # extract patch
            patch = orthomosaic_utils.get_patch(img=orthomosaic, size=patch_size, top_left_coord=(x, y))
            # print(f'original patch.shape: {patch.shape}')

            # apply preprocessing
            if preprocess is not None:
                patch = preprocess(patch)

            # to device
            patch = patch.to(device)

            single_patch_batch = patch.unsqueeze(0)
            # print(f'single_patch_batch.shape: {single_patch_batch.shape}')
            single_prediction_batch = loaded_model(single_patch_batch)
            # print(f'single_prediction_batch.shape: {single_prediction_batch.shape}')
            prediction = single_prediction_batch.squeeze(0)
            # print(f'prediction.shape: {prediction.shape}')
            # print(f'prediction: {prediction}')
            prediction = softmax(prediction)
            # print(f'prediction softmax: {prediction}')
            prediction = prediction.detach().cpu().numpy()
            # print(f'prediction.detach().cpu().numpy(): {prediction}')
            predicted_class = prediction.argmax()
            # print(f'predicted_class: {predicted_class}')
            predicted_probability = prediction[predicted_class]
            # print(f'predicted_probability: {predicted_probability}')
            if predicted_probability < confidence_threshold:
                predicted_class = unknown_class_id
                # print('predicted_class unknown')

            species_distribution[
                x : x + patch_size,
                y : y + patch_size,
                predicted_class,
            ] += 1
            species_distribution[
                x : x + patch_size,
                y : y + patch_size,
                -1,
            ] += 1

    end_time = datetime.datetime.now()
    print(f'loop time: {utils.timedelta_format(start_time, end_time)}')

    # create and save one image for each species
    start_time = end_time
    print(f'species_distribution.shape: {species_distribution.shape}')
    effective_max_x = x + patch_size
    effective_max_y = y + patch_size
    print(f'effective_max_x: {effective_max_x}, effective_max_y: {effective_max_y}')
    species_distribution = species_distribution[ : effective_max_x, : effective_max_y, : ]
    print(f'species_distribution.shape: {species_distribution.shape}')

    # without last channel/layer (prediction count)
    temp_species_distribution = np.zeros(shape=species_distribution[ : , : , : -1].shape, dtype=np.float32)
    print(f'temp_species_distribution.shape: {temp_species_distribution.shape}')
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        for class_index in range(num_classes_plus_unknown):
            temp_species_distribution[ : , : , class_index] = species_distribution[: , : , class_index] / species_distribution[: , : , -1]
    np.nan_to_num(temp_species_distribution, copy=False, nan=0.0)
    species_distribution = temp_species_distribution
    # print('Unique values count')
    # for species_index in range(num_classes_plus_unknown):
    #     print(f'- {global_constants.TREE_INFORMATION[species_index][global_constants.SPECIES_LANGUAGE]}:'
    #           f' {np.unique(species_distribution[ : , : , species_index], return_counts=True)}')

    assert species_distribution.max() <= 1, f'species_distribution.max() > 1. (max = {species_distribution.max()})'
    assert species_distribution.min() >= 0, f'species_distribution.min() < 0. (min = {species_distribution.min()})'

    save_path = f'{global_constants.OUTPUT_DIR}{global_constants.ORTHOMOSAIC_FOLDER_NAME}{img_name_no_extension}/'
    orthomosaic_utils.save_output(
        num_classes_plus_unknown=num_classes_plus_unknown,
        unknown_class_id=unknown_class_id,
        species_distribution=species_distribution,
        save_path=save_path,
        effective_max_x=effective_max_x,
        effective_max_y=effective_max_y,
        orthomosaic=orthomosaic,
    )
    end_time = datetime.datetime.now()
    print(f'species image creation and saving time: {utils.timedelta_format(start_time, end_time)}')
    print(f'TOTAL TIME: {utils.timedelta_format(global_start_time, end_time)}')


if __name__ == '__main__':
    verbose = 2
    partial_name = str(input('Insert name or part of the name of a model: '))
    model_id = int(input('Insert model id number: '))
    # img_name = str(input('Insert name of the orthomosaic to analyse: '))
    img_name = 'Zao1_211005.tif'
    analyse_orthomosaic_(partial_name=partial_name, model_id=model_id, img_name=img_name, verbose=verbose)
