import copy
import numpy as np
import torch
import torchmetrics

import config
import global_constants
from models import model_utils


def train(model: torch.nn.Module,
          training_data,
          validation_data,
          loss_function_name: str,
          optimizer_name: str,
          learning_rate: float,
          device: torch.device,
          num_epochs: int,
          save_model: bool,
          metrics: dict = None,
          save_path=None,
          custom_transforms=None,
          verbose: int = 0,
          ) -> dict:

    if verbose >= 1:
        print('Training started...')

    # get loss function from string name
    loss_function = getattr(torch.nn, loss_function_name)()

    # get optimizer from string name
    optimizer = getattr(torch.optim, optimizer_name)(params=model.parameters(), lr=learning_rate)
    classes = []
    for tree_info in global_constants.TREE_INFORMATION.values():
        classes.append(tree_info['japanese_reading'])

    training_metrics = {}
    validation_metrics = {}
    num_classes = len(global_constants.TREE_INFORMATION)
    if metrics is None:
        metrics = {}
    for metric_name, metric_args in metrics.items():
        try:
            metric_class = getattr(torchmetrics, metric_name)
        except AttributeError:
            raise AttributeError(f'metric {metric_name} not found in torchmetrics')

        metric_args['num_classes'] = num_classes
        training_metrics[metric_name] = metric_class(**metric_args)
        validation_metrics[metric_name] = metric_class(**metric_args)

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
            history['metrics']['train'][metric_name] = metric.compute().item()
        for metric_name in validation_metrics:
            metric = validation_metrics[metric_name]
            history['metrics']['validation'][metric_name] = metric.compute().item()

        if verbose >= 1:
            model_utils.print_formatted_results(
                title='TRAINING RESULTS',
                loss=history['loss']['train'][-1],
                metrics=history['metrics']['train'],
                metrics_in_percentage=True,
            )

            model_utils.print_formatted_results(
                title='VALIDATION RESULTS',
                loss=history['loss']['validation'][-1],
                metrics=history['metrics']['validation'],
                metrics_in_percentage=True,
            )

    # if training is interrupted, save the best model obtained so far
    except KeyboardInterrupt:
        print('Training interrupted')
        if save_model and best_model_weights is not None:
            print('Saving model before exiting...')
            print(f'With validation loss: {min_valid_loss}')
            model.load_state_dict(best_model_weights)
            meta_data = {
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'loss_function_name': loss_function_name,
                'optimizer_name': optimizer_name,
                'training_length': len(training_data),
                'interrupted': True,
                'shuffle': config.SHUFFLE,
                'random_seed': config.RANDOM_SEED,
                'batch_size': batch_size,
                'num_classes': num_classes,
                'classes': classes,
                'train_val_test_proportions': config.TRAIN_VAL_TEST_PROPORTIONS,
                'last_complete_epoch': epoch - 1,
                'augmentation_proportion': config.DATA_AUGMENTATION_PROPORTION,
                'balance_classes': config.BALANCE_DATA,
                'history': history,
            }
            model_utils.save_model_and_meta_data(
                model=model,
                custom_transforms=custom_transforms,
                save_path=save_path,
                meta_data=meta_data,
                verbose=verbose,
            )
        raise KeyboardInterrupt

    # save model (weights and configuration) and other useful information
    if save_model:
        meta_data = {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'loss_function_name': loss_function_name,
            'optimizer_name': optimizer_name,
            'training_length': len(training_data),
            'interrupted': False,
            'shuffle': config.SHUFFLE,
            'random_seed': config.RANDOM_SEED,
            'batch_size': batch_size,
            'num_classes': num_classes,
            'classes': classes,
            'train_val_test_proportions': config.TRAIN_VAL_TEST_PROPORTIONS,
            'last_complete_epoch': epoch,
            'augmentation_proportion': config.DATA_AUGMENTATION_PROPORTION,
            'balance_classes': config.BALANCE_DATA,
            'history': history,
        }
        model_utils.save_model_and_meta_data(
            model=model,
            custom_transforms=custom_transforms,
            save_path=save_path,
            meta_data=meta_data,
            verbose=verbose,
        )
    return history
