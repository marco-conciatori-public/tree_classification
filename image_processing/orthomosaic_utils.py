import json
import torch
import numpy as np
from PIL import Image
import tifffile as tifi
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import global_constants
import utils


def get_patch(img: torch.Tensor, size: int, top_left_coord: tuple):
    # for channel_first img
    x = top_left_coord[0]
    y = top_left_coord[1]
    patch = img[
            :,
            x : x + size,
            y : y + size,
    ].detach().clone()
    return patch


def save_class_layers(num_classes_plus_unknown: int,
                      unknown_class_id: int,
                      species_distribution: np.ndarray,
                      save_path: str,
                      class_information: dict,
                      ):
    # create one image for each class, where the alpha channel is the probability of that pixel being that class
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for class_index in range(num_classes_plus_unknown):
        temp_img = np.zeros((species_distribution.shape[0], species_distribution.shape[1], 4), dtype=np.uint8)
        temp_img[:, :, 3] = species_distribution[:, :, class_index] * 255
        color = (0, 0, 0)
        class_name = 'unknown'
        if class_index != unknown_class_id:
            color = class_information[class_index]['display_color_rgb']
            class_name = class_information[class_index][global_constants.SPECIES_LANGUAGE]
        temp_img[:, :, 0] = color[0]
        temp_img[:, :, 1] = color[1]
        temp_img[:, :, 2] = color[2]

        tifi.imwrite(f'{save_path}{class_name}.tif', data=temp_img)


def save_effective_orthomosaic(orthomosaic: torch.Tensor, effective_max_x: int, effective_max_y: int, save_path: str):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    print(f'orthomosaic.shape: {orthomosaic.shape}')
    orthomosaic = orthomosaic.permute(1, 2, 0).numpy()
    print(f'orthomosaic shape colors last: {orthomosaic.shape}')
    orthomosaic = orthomosaic[: effective_max_x, : effective_max_y, :]
    print(f'orthomosaic effective shape: {orthomosaic.shape}')
    tifi.imwrite(
        f'{save_path}subset_img.tif',
        data=orthomosaic,
    )


def save_legend(save_path: str,
                num_classes_plus_unknown: int,
                unknown_class_id: int,
                class_information: dict,
                expand=(0, 0, 0, 0),
                ):
    # create a legend with the colors of the classes
    patches = []
    for class_id in range(num_classes_plus_unknown):
        color = [0, 0, 0]
        name = 'unknown'
        if class_id != unknown_class_id:
            name = class_information[class_id][global_constants.SPECIES_LANGUAGE]
            color = list(class_information[class_id]['display_color_rgb'])
            for rgb_index in range(3):
                color[rgb_index] = color[rgb_index] / 255
        patches.append(mpatches.Patch(color=color, label=name))
    legend = plt.legend(handles=patches)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f'{save_path}legend.png', dpi="figure", bbox_inches=bbox)
    plt.close()


def save_output(orthomosaic: torch.Tensor,
                species_distribution: np.ndarray,
                save_path: str,
                info: dict,
                ):

    save_class_layers(
        num_classes_plus_unknown=info['num_classes_plus_unknown'],
        unknown_class_id=info['unknown_class_id'],
        species_distribution=species_distribution,
        save_path=save_path,
        class_information=info['model_meta_data']['class_information'],
    )
    save_effective_orthomosaic(
        orthomosaic=orthomosaic,
        effective_max_x=info['effective_max_x'],
        effective_max_y=info['effective_max_y'],
        save_path=save_path,
    )
    save_legend(
        save_path=save_path,
        num_classes_plus_unknown=info['num_classes_plus_unknown'],
        unknown_class_id=info['unknown_class_id'],
        class_information=info['model_meta_data']['class_information'],
    )
    with open(f'{save_path}{global_constants.INFO_FILE_NAME}.json', 'w') as file:
        json.dump(info, file, indent=4)


def load_target(folder_path: str, info: dict, target_extension: str = '', verbose: int = 0) -> dict:
    # load all the target images in the folder
    target_dict = {}
    shape = None
    for file in Path(folder_path).rglob('*' + target_extension):
        if verbose >= 2:
            print(f'loading target: {file}')
        if 'orthomosaic' in file.name:
            print('\tskipping orthomosaic')
            continue
        # the name of the file is the species name, after removing file extension
        species_name = file.name.split('.')[0]
        print(f'species_name: {species_name}')
        species_id = utils.get_species_id_by_name(species_name, info['model_meta_data']['class_information'])
        img = load_img(file)
        # limit the size of the target to the effective size of the orthomosaic
        img = img[: info['effective_max_x'], : info['effective_max_y'], :]
        if species_id != -1:
            target_dict[species_id] = img
            if shape is None:
                shape = img.shape
            else:
                if shape != img.shape:
                    raise ValueError(f'targets have different shapes: {shape} and {target_dict[species_id].shape}')
        else:
            if verbose >= 1:
                print(f'skipping species {species_name}: it is not among the ones the model is trained to recognise')
    if verbose >= 1:
        print(f'loaded {len(target_dict)} targets')
    return target_dict


def evaluate_results(prediction: np.array, target: dict, info: dict, verbose: int = 0) -> dict:
    evaluation = {}
    evaluation['total'] = {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'accuracy': 0,
        }
    # targets are images with the same shape as the predictions
    # and are black where the class is present and white where it is not
    for species_index in range(info['num_classes_plus_unknown']):
        if species_index == info['unknown_class_id']:
            continue
        if verbose >= 2:
            print(f'evaluating species: {info["model_meta_data"]["class_information"][species_index][global_constants.SPECIES_LANGUAGE]}')

        # targets and predictions are in different dimensions in the arrays
        species_target = target[species_index]
        species_prediction = prediction[:, :, species_index]
        # print(f'species_prediction.shape: {species_prediction.shape}')
        # print(f'species_target.shape: {species_target.shape}')
        # print(f'species_target[2000][2000]: {species_target[2000][2000]}')
        species_target = species_target.sum(axis=2)
        species_target = species_target / (255 * 3)

        # create auxiliary arrays to calculate the true positives, false positives, true negatives and false negatives
        species_target_bool = np.equal(species_target, 0)
        species_prediction_bool = np.logical_not(np.equal(species_prediction, 0))
        # print(f'species_target_bool.shape: {species_target_bool.shape}')
        # print(f'species_prediction_bool.shape: {species_prediction_bool.shape}')
        # print(f'species_target_bool[2000][2000]: {species_target_bool[2000][2000]}')
        # print(f'species_prediction_bool[2000][2000]: {species_prediction_bool[2000][2000]}')
        # temp_img = np.zeros((species_target_bool.shape[0], species_target_bool.shape[1], 4), dtype=np.uint8)
        # temp_img[:, :, 3] = species_target_bool * 255
        # color = info['model_meta_data']['class_information'][species_index]['display_color_rgb']
        # temp_img[:, :, 0] = color[0]
        # temp_img[:, :, 1] = color[1]
        # temp_img[:, :, 2] = color[2]
        # tifi.imwrite('species_target.tif', data=temp_img)
        # temp_img[:, :, 3] = species_prediction_bool * 255
        # tifi.imwrite('species_prediction.tif', data=temp_img)

        # calculate the true positives, false positives, true negatives and false negatives
        species_target_bool_negated = np.logical_not(species_target_bool)
        species_prediction_bool_negated = np.logical_not(species_prediction_bool)
        true_positives = np.sum(np.logical_and(species_target_bool, species_prediction_bool))
        false_positives = np.sum(np.logical_and(species_target_bool_negated, species_prediction_bool))
        true_negatives = np.sum(np.logical_and(species_target_bool_negated, species_prediction_bool_negated))
        false_negatives = np.sum(np.logical_and(species_target_bool, species_prediction_bool_negated))
        print(f'true_positives: {true_positives}')
        print(f'false_positives: {false_positives}')
        print(f'true_negatives: {true_negatives}')
        print(f'false_negatives: {false_negatives}')

        # calculate the precision, recall and f1 score
        # ifs are to avoid division by zero
        precision = 0
        recall = 0
        f1_score = 0
        accuracy = 0
        if true_positives > 0:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
        if (precision > 0) and (recall > 0):
            f1_score = 2 * precision * recall / (precision + recall)
        if (true_positives + true_negatives) > 0:
            accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
        if verbose >= 2:
            print(f'\tprecision: {precision}')
            print(f'\trecall: {recall}')
            print(f'\tf1_score: {f1_score}')
            print(f'\taccuracy: {accuracy}')
        evaluation[info["model_meta_data"]["class_information"][species_index][global_constants.SPECIES_LANGUAGE]] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
        }
        evaluation['total']['precision'] += precision
        evaluation['total']['recall'] += recall
        evaluation['total']['f1_score'] += f1_score
        evaluation['total']['accuracy'] += accuracy
    num_classes = info['model_meta_data']['num_classes']
    evaluation['total']['precision'] /= num_classes
    evaluation['total']['recall'] /= num_classes
    evaluation['total']['f1_score'] /= num_classes
    evaluation['total']['accuracy'] /= num_classes
    return evaluation


def load_img(orthomosaic_path) -> np.array:
    try:
        return tifi.imread(orthomosaic_path)
    except Exception:
        pass
    try:
        return np.array(Image.open(orthomosaic_path))
    except Exception:
        raise ValueError(f'could not load image: {orthomosaic_path}')
