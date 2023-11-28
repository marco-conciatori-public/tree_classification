import copy
import torch
import numpy as np

import utils
import global_constants
from models import model_utils


def train(model: torch.nn.Module,
          training_data,
          validation_data,
          loss_function_name: str,
          optimizer_parameters: dict,
          device: torch.device,
          num_epochs: int,
          save_model: bool,
          metrics: dict = None,
          save_path=None,
          custom_transforms=None,
          verbose: int = 0,
          extra_info_to_save: dict = None,
          ) -> dict:

    if verbose >= 1:
        print('Training started...')

    # get loss function from string name
    loss_function = getattr(torch.nn, loss_function_name)()

    # get optimizer from string name
    # TODO: if freeze_layers is True, the optimizer should only take as parameters the parameters of the last layer/s
    optimizer = getattr(torch.optim, optimizer_parameters['optimizer_name'])(
        params=model.parameters(),
        lr=optimizer_parameters['learning_rate'],
        weight_decay=optimizer_parameters['weight_decay'],
    )
    classes = []
    for tree_info in global_constants.TREE_INFORMATION.values():
        classes.append(tree_info[global_constants.SPECIES_LANGUAGE])

    num_classes = len(global_constants.TREE_INFORMATION)
    training_metrics = model_utils.get_metrics(metrics)
    validation_metrics = model_utils.get_metrics(metrics)

    history = {
        'loss': {},
        'metrics': {}
    }
    history['loss']['train'] = []
    history['loss']['validation'] = []
    history['metrics']['train'] = {}
    history['metrics']['validation'] = {}

    try:
        # loop over the dataset 'num_epochs' times
        best_model_weights = None
        min_valid_loss = np.inf
        batch_size = 0
        for epoch in range(num_epochs):
            batch_counter = 0
            # TRAINING
            training_loss = 0.0
            model.train()
            for batch in training_data:
                observation_batch, target_batch = batch

                if batch_counter == 0:
                    batch_size = len(target_batch)

                if not target_batch.shape[0] == batch_size:
                    continue

                # Potentially transfer batch to GPU
                if observation_batch.device != device:
                    observation_batch = observation_batch.to(device)
                    target_batch = target_batch.to(device)
                # print(f'observation_batch.shape: {observation_batch.shape}')
                # print(f'target_batch.shape: {target_batch.shape}')
                # print(f'target_batch: {target_batch}')

                prediction_batch = model(observation_batch)
                # print(f'prediction_batch.shape: {prediction_batch.shape}')
                loss = loss_function(prediction_batch, target_batch)

                # update metrics
                for metric in training_metrics.values():
                    metric.update(prediction_batch.cpu(), target_batch.cpu())

                optimizer.zero_grad()
                loss.backward()
                # update network parameters
                optimizer.step()

                training_loss += loss.item()
                batch_counter += 1

            # VALIDATION
            validation_loss = 0.0
            batch_counter = 0
            model.eval()
            with torch.set_grad_enabled(False):
                for batch in validation_data:
                    observation_batch, target_batch = batch

                    if batch_counter == 0:
                        batch_size = len(target_batch)

                    if not target_batch.shape[0] == batch_size:
                        continue

                    # Potentially transfer batch to GPU
                    if observation_batch.device != device:
                        observation_batch = observation_batch.to(device)
                        target_batch = target_batch.to(device)

                    prediction_batch = model(observation_batch)
                    loss = loss_function(prediction_batch, target_batch)

                    # update metrics
                    for metric in validation_metrics.values():
                        metric.update(prediction_batch.cpu(), target_batch.cpu())

                    validation_loss += loss.item()
                    batch_counter += 1

            training_loss = training_loss / len(training_data)
            validation_loss = validation_loss / len(validation_data)

            if verbose >= 2:
                print(f'Epoch {epoch + 1}')

                # print metrics results for the current epoch
                training_results = utils.get_metric_results(training_metrics, metrics)
                validation_results = utils.get_metric_results(validation_metrics, metrics)
                model_utils.print_formatted_results(
                    title='TRAINING RESULTS',
                    loss=training_loss,
                    metrics=training_results,
                )
                model_utils.print_formatted_results(
                    title='VALIDATION RESULTS',
                    loss=validation_loss,
                    metrics=validation_results,
                )

            history['loss']['train'].append(training_loss)
            history['loss']['validation'].append(validation_loss)

            if validation_loss < min_valid_loss:
                if verbose >= 2:
                    print(f' \t\t Validation loss decreased({min_valid_loss:.6f} '
                          f'--> {validation_loss:.6f}) \t Saving the model')

                min_valid_loss = validation_loss
                # Saving model weights
                best_model_weights = copy.deepcopy(model.state_dict())

        # reload the best model found
        model.load_state_dict(best_model_weights)

        history['metrics']['train'] = utils.get_metric_results(training_metrics, metrics)
        history['metrics']['validation'] = utils.get_metric_results(validation_metrics, metrics)
        if verbose >= 1:
            model_utils.print_formatted_results(
                title='TRAINING RESULTS',
                loss=history['loss']['train'][-1],
                metrics=history['metrics']['train'],
            )
            model_utils.print_formatted_results(
                title='VALIDATION RESULTS',
                loss=history['loss']['validation'][-1],
                metrics=history['metrics']['validation'],
            )

    # if training is interrupted, save the best model obtained so far
    except KeyboardInterrupt:
        print('Training interrupted')
        if save_model and best_model_weights is not None:
            print('Saving model before exiting...')
            print(f'With validation loss: {min_valid_loss}')
            model.load_state_dict(best_model_weights)
            extra_info_to_save['num_epochs'] = num_epochs
            extra_info_to_save['loss_function_name'] = loss_function_name
            extra_info_to_save['optimizer_parameters'] = optimizer_parameters
            extra_info_to_save['training_length'] = len(training_data)
            extra_info_to_save['interrupted'] = True
            extra_info_to_save['batch_size'] = batch_size
            extra_info_to_save['num_classes'] = num_classes
            extra_info_to_save['classes'] = classes
            extra_info_to_save['last_complete_epoch'] = epoch - 1
            extra_info_to_save['history'] = history

            model_utils.save_model_and_meta_data(
                model=model,
                custom_transforms=custom_transforms,
                save_path=save_path,
                meta_data=extra_info_to_save,
                verbose=verbose,
            )
        raise KeyboardInterrupt

    # save model (weights and configuration) and other useful information
    if save_model:
        extra_info_to_save['num_epochs'] = num_epochs
        extra_info_to_save['loss_function_name'] = loss_function_name
        extra_info_to_save['optimizer_parameters'] = optimizer_parameters
        extra_info_to_save['training_length'] = len(training_data)
        extra_info_to_save['interrupted'] = False
        extra_info_to_save['batch_size'] = batch_size
        extra_info_to_save['num_classes'] = num_classes
        extra_info_to_save['classes'] = classes
        extra_info_to_save['last_complete_epoch'] = epoch
        extra_info_to_save['history'] = history

        model_utils.save_model_and_meta_data(
            model=model,
            custom_transforms=custom_transforms,
            save_path=save_path,
            meta_data=extra_info_to_save,
            verbose=verbose,
        )
    return history
