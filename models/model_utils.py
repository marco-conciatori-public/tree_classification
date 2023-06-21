import json
import copy
import torch
from pathlib import Path

import utils
import global_constants
from models import conv_2d


def create_model(model_class_name: str,
                 input_shape: tuple,
                 num_output: int,
                 model_parameters: dict,
                 device: torch.device,
                 ) -> torch.nn.Module:
    assert device is not None, f'ERROR: torch could not find suitable cpu or gpu to use (device: {device})'

    try:
        name = model_parameters['name']
        if name is None:
            name = model_class_name
    except KeyError:
        name = model_class_name

    model_id = utils.get_available_id(partial_name=name, folder_path=global_constants.MODEL_OUTPUT_DIR)
    # warning: parallel or concurrent model instances it is possible that some of them get assigned the same id.
    # The error or involuntary overwriting happens only when saving those models.

    if model_class_name == 'Conv_2d':
        model = conv_2d.Conv_2d(
            input_shape=input_shape,
            num_output=num_output,
            name=name,
            model_id=model_id,
            **model_parameters,
        )

    else:
        raise NotImplementedError(f'ERROR: the model class {model_class_name} is not implemented.')

    # migrate model to cpu or gpu depending on the available device
    model.to(device=device)

    # convert model to float64 dtype
    # model.double()
    return model


def save_model_and_meta_data(model: torch.nn.Module,
                             meta_data: dict,
                             path,
                             learning_rate: float,
                             epochs: int,
                             loss_function_name: str,
                             optimizer_name: str,
                             # loss=None,
                             verbose: int = 0,
                             ):

    if verbose >= 1:
        print('Saving model...')
    Path(path).mkdir(parents=True, exist_ok=True)
    file_name = f'{model.name}{global_constants.EXTERNAL_PARAMETER_SEPARATOR}{model.id}'

    assert file_name is not None, 'ERROR: unable to find model file name information.'

    complete_path = path + file_name + global_constants.PYTORCH_FILE_EXTENTION
    torch.save(model, complete_path)

    meta_data_to_save = copy.deepcopy(meta_data)
    meta_data_to_save['training_epochs'] = epochs
    meta_data_to_save['initial_learning_rate'] = learning_rate
    meta_data_to_save['loss_function_name'] = loss_function_name
    meta_data_to_save['optimizer_name'] = optimizer_name
    meta_data_to_save['model_layers'] = model.layers

    with open(
            f'{path}{file_name}{global_constants.EXTERNAL_PARAMETER_SEPARATOR}'
            f'{global_constants.INFO_FILE_NAME}.json',
            'w'
    ) as json_file:
        json.dump(meta_data_to_save, json_file, default=str)

    if verbose >= 1:
        print(f'Model saved successfully ({complete_path}).')


def load_model(model_path: str,
               device: torch.device = None,
               training_mode: bool = False,
               meta_data_path=None,
               verbose: int = 0,
               ) -> (torch.nn.Module, dict):
    if device is None:
        device = utils.get_available_device(verbose=verbose)
    model = torch.load(model_path, map_location=device)
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
    return model, meta_data


def print_formatted_results(loss: float, metrics: dict, title: str = 'RESULTS'):
    print(title)
    print(f'- Loss: {loss}')
    for metric_name in metrics:
        result = metrics[metric_name]
        try:
            content_result = result.item()
        except AttributeError:
            content_result = result
        print(f'- {metric_name}: {round(content_result, global_constants.MAX_DECIMAL_PLACES)}')
