import copy
import numpy as np
import torch
import torchmetrics
import ignite.metrics

import global_constants
from models import model_utils


def train(model: torch.nn.Module,
          train_data,
          validation_data,
          loss_function_name: str,
          optimizer_name: str,
          learning_rate: float,
          device: torch.device,
          epochs: int,
          save_model: bool,
          meta_data: dict = None,
          metrics=(),
          save_path=None,
          verbose: int = 0,
          ):

    # get loss function from string name
    loss_function = getattr(torch.nn, loss_function_name)()

    # get optimizer from string name
    optimizer = getattr(torch.optim, optimizer_name)(params=model.parameters(), lr=learning_rate)

    training_metrics = {}
    validation_metrics = {}
    for metric_name in metrics:
        try:
            metric_class = getattr(torchmetrics, metric_name)
        except AttributeError:
            try:
                metric_class = getattr(ignite.metrics, metric_name)
            except AttributeError:
                raise AttributeError(f'metric {metric_name} not found in custom metrics, torchmetrics or ignite.')

        training_metric_instance = metric_class()
        validation_metric_instance = metric_class()

        training_metrics[metric_name] = training_metric_instance
        validation_metrics[metric_name] = validation_metric_instance

    history = {
        'loss': {},
        'metrics': {}
    }
    history['loss']['train'] = []
    history['loss']['validation'] = []
    history['metrics']['train'] = {}
    history['metrics']['validation'] = {}

    # loop over the dataset 'epochs' times
    best_model_weights = None
    min_valid_loss = np.inf
    batch_size = 0
    for epoch in range(epochs):
        batch_counter = 0
        # TRAINING
        training_loss = 0.0
        model.train()
        for batch in train_data:
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

        training_loss = training_loss / len(train_data)
        validation_loss = validation_loss / len(validation_data)

        if verbose >= 2:
            print(f'Epoch {epoch + 1}')

            # print metrics results for the current epoch
            print('Training results:')
            print(f'\tTraining Loss: {training_loss}')
            for metric_name in training_metrics:
                metric = training_metrics[metric_name]
                result = metric.compute()
                print(f'\t{metric_name}: {round(result.item(), global_constants.MAX_DECIMAL_PLACES)}')

            print('Validation results:')
            print(f'\tValidation Loss: {validation_loss}')
            for metric_name in validation_metrics:
                metric = validation_metrics[metric_name]
                result = metric.compute()
                print(f'\t{metric_name}: {round(result.item(), global_constants.MAX_DECIMAL_PLACES)}')

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

    for metric_name in training_metrics:
        metric = training_metrics[metric_name]
        history['metrics']['train'][metric_name] = metric.compute()
    for metric_name in validation_metrics:
        metric = validation_metrics[metric_name]
        history['metrics']['validation'][metric_name] = metric.compute()

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

    # save model (weights and configuration) and other useful information
    if save_model:
        model_utils.save_model_and_meta_data(
            model=model,
            meta_data=meta_data,
            path=save_path,
            learning_rate=learning_rate,
            epochs=epochs,
            loss_function_name=loss_function_name,
            optimizer_name=optimizer_name,
            # loss={
            #    'train': history['loss']['train'][-1],
            #    'validation': history['loss']['validation'][-1]
            #    },
            verbose=verbose,
        )
    return history