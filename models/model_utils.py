import json
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
                             meta_data: dict = None,
                             verbose: int = 0,
                             ):

    if verbose >= 1:
        print('Saving model...')
    Path(save_path).mkdir(parents=True, exist_ok=True)

    file_name = f'{model.name}{global_constants.EXTERNAL_PARAMETER_SEPARATOR}{model.id}'
    assert file_name is not None, 'ERROR: unable to retrieve model information'

    model_path = save_path + file_name + global_constants.PYTORCH_FILE_EXTENSION
    torch.save(model, model_path)
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
