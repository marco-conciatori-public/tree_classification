import json
import torch
import warnings
from pathlib import Path
from torchvision import models
import torchvision.transforms.functional as tf

import utils
import global_constants
from models import conv_2d


def create_model(model_class_name: str,
                 input_shape: tuple,
                 num_output: int,
                 model_parameters: dict,
                 device: torch.device,
                 name: str = None,
                 verbose: int = 0,
                 ) -> torch.nn.Module:
    assert device is not None, f'ERROR: torch could not find suitable cpu or gpu to use (device: {device})'

    if name is None:
        name = model_class_name

    model_id = utils.get_available_id(partial_name=name, folder_path=global_constants.MODEL_OUTPUT_DIR)
    # warning: parallel or concurrent model instances it is possible that some of them get assigned the same id.
    # The error or involuntary overwriting happens only when saving those models.

    if verbose >= 2:
        print(f'Creating model {name} with id {model_id}...')
        # print(f'Input shape: {input_shape}')
        # print(f'Number of output classes: {num_output}')
        # print(f'Model parameters: {model_parameters}')

    if model_class_name == 'Conv_2d':
        model = conv_2d.Conv_2d(
            input_shape=input_shape,
            num_output=num_output,
            name=name,
            model_id=model_id,
            **model_parameters,
        )

    else:
        raise NotImplementedError(f'ERROR: the model class {model_class_name} is not implemented')

    # migrate model to cpu or gpu depending on the available device
    model.to(device=device)
    return model


def save_model_and_meta_data(model: torch.nn.Module,
                             save_path: str,
                             custom_transforms=(),
                             meta_data: dict = None,
                             verbose: int = 0,
                             ):

    if verbose >= 1:
        print('Saving model...')
    Path(save_path).mkdir(parents=True, exist_ok=True)

    file_name = f'{model.name}{global_constants.EXTERNAL_PARAMETER_SEPARATOR}{model.id}'
    assert file_name is not None, 'ERROR: unable to retrieve model information'

    model_path = save_path + file_name + global_constants.PYTORCH_FILE_EXTENSION
    torch.save(obj=(model, custom_transforms), f=model_path)
    if verbose >= 1:
        print(f'Model saved successfully ({model_path})')

    if meta_data is not None:
        meta_data_path = f'{save_path}{file_name}{global_constants.EXTERNAL_PARAMETER_SEPARATOR}' \
                         f'{global_constants.INFO_FILE_NAME}.json'
        with open(file=meta_data_path, mode='w') as json_file:
            json.dump(meta_data, json_file, default=str)
        if verbose >= 1:
            print(f'Meta data saved successfully ({meta_data_path})')


def load_model(model_path: str,
               device: torch.device = None,
               training_mode: bool = False,
               meta_data_path=None,
               verbose: int = 0,
               ) -> (torch.nn.Module, dict):
    if device is None:
        device = utils.get_available_device(verbose=verbose)
    model, custom_transforms = torch.load(model_path, map_location=device)
    # normally, we save trained model, so we want to test or use them, not train them again
    if not training_mode:
        model.eval()
    else:
        model.train()

    meta_data = {}
    if meta_data_path is not None:
        with open(meta_data_path, 'r') as json_file:
            meta_data = json.load(json_file)

    if verbose >= 1:
        print(f'model name: {model.name}')
    return model, custom_transforms, meta_data


def print_formatted_results(loss: float,
                            metrics: dict,
                            metrics_in_percentage: bool = False,
                            title: str = 'RESULTS'
                            ):
    print(title)
    print(f'- Loss: {loss}')
    for metric_name in metrics:
        result = metrics[metric_name]
        if metrics_in_percentage:
            print(f'- {metric_name}: {round(result * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))} %')
        else:
            print(f'- {metric_name}: {round(result, global_constants.MAX_DECIMAL_PLACES)}')


def get_torchvision_model(model_name: str,
                          device: torch.device,
                          use_new_weights: bool = False,
                          weights_name: str = None,
                          training: bool = False,
                          num_classes: int = None,
                          verbose: int = 0,
                          ) -> torch.nn.Module:
    if verbose >= 1:
        if use_new_weights and weights_name is not None:
            warnings.warn(f'use_new_weights is True, but weights_name {weights_name} is given. weights_name will'
                          f' be ignored and new weights will be used')

    if weights_name is None and not use_new_weights:
        raise ValueError('weights_name must be specified when use_new_weights = False. Otherwise, use_new_weights'
                         ' must be set to True')

    model_full_name = f'{model_name}{global_constants.INTERNAL_PARAMETER_SEPARATOR}Weights' \
                      f'{global_constants.EXTERNAL_PARAMETER_SEPARATOR}{weights_name}'
    model_id = utils.get_available_id(partial_name=model_full_name, folder_path=global_constants.MODEL_OUTPUT_DIR)

    # Initialize model with the given weights
    # if weights_name == 'DEFAULT':
    #     weights_name = f'{model_name}_Weights.DEFAULT'
    weights = models.get_weight(name=weights_name)
    model = models.get_model(name=model_name.lower(), weights=weights)
    model.name = model_full_name
    model.id = model_id

    if training:
        assert num_classes is not None, 'num_classes must be specified when training = True'
        model.train()

        # Freeze all layers weights
        for param in model.parameters():
            param.requires_grad = False
        # Replace the last layer with a new, untrained layer that has num_classes outputs
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        # unfreeze the last layer
        model.fc.requires_grad = True

    else:
        model.eval()
        # Freeze all layers weights
        for param in model.parameters():
            param.requires_grad = False

    # print(f'model:\n{model}')
    model.to(device=device)
    return model


def get_custom_transforms(weights_name: str,
                          verbose: int = 0,
                          ):

    weights = models.get_weight(name=weights_name)
    preprocess =weights.transforms(antialias=True)
    attributes = dir(preprocess)
    resize_in_attributes = False
    for attribute in attributes:
        if 'resize' in attribute.lower():
            resize_in_attributes = True
            break

    custom_transforms = [
        tf.to_tensor,
        preprocess,
    ]
    if verbose >= 2:
        print(f'resize_in_attributes: {resize_in_attributes}')
        print(f'custom_transforms: {custom_transforms}')
    return custom_transforms, resize_in_attributes


def save_test_results(cm_true_values: list,
                      cm_predictions: list,
                      model: torch.nn.Module,
                      save_path: str,
                      test_loss: float,
                      metric_evaluations: dict,
                      verbose: int = 0,
                      ):
    if verbose >= 1:
        print(f'Updating meta data information with test results...')

    file_name = f'{model.name}{global_constants.EXTERNAL_PARAMETER_SEPARATOR}{model.id}'
    assert file_name is not None, 'ERROR: unable to retrieve model information'
    meta_data_path = f'{save_path}{file_name}{global_constants.EXTERNAL_PARAMETER_SEPARATOR}' \
                     f'{global_constants.INFO_FILE_NAME}.json'
    if verbose >= 2:
        print(f'meta data path: "{meta_data_path}"')

    with open(meta_data_path, 'r') as json_file:
        meta_data = json.load(json_file)
        meta_data['history']['loss']['test'] = [test_loss]
        meta_data['history']['metrics']['test'] = {}
        for metric_name in metric_evaluations:
            meta_data['history']['metrics']['test'][metric_name] = metric_evaluations[metric_name]
        meta_data['test_confusion_matrix'] = {}
        meta_data['test_confusion_matrix']['true_values'] = cm_true_values
        meta_data['test_confusion_matrix']['predictions'] = cm_predictions

    with open(meta_data_path, 'w') as json_file:
        json.dump(meta_data, json_file, default=str)
    if verbose >= 1:
        print(f'Meta data updated successfully')
