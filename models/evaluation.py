import torch

from models import model_utils
from metrics import metric_utils
from visualization import visualization_utils
from metrics.metric_manager import MetricManager


def eval(model: torch.nn.Module,
         test_data,
         loss_function_name: str,
         device: torch.device,
         class_information: dict,
         metrics: dict = None,
         display_confusion_matrix: bool = False,
         save_results: bool = False,
         save_path=None,
         verbose: int = 0,
         ) -> (float, dict):

    if verbose >= 1:
        print('Evaluation started...')

    # get loss function from string name
    loss_function = getattr(torch.nn, loss_function_name)()
    test_metric_manager = MetricManager(
        biodiversity_metric_names=metrics['biodiversity'],
        classification_metric_names=metrics['classification'],
        class_information=class_information,
    )

    test_loss = 0.0
    batch_counter = 0
    model.eval()
    tag_list = []
    prediction_list = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.set_grad_enabled(False):
        for batch in test_data:
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

            prediction_batch = prediction_batch.cpu()
            target_batch = target_batch.cpu()
            # update metrics
            test_metric_manager.update(predicted_probabilities=prediction_batch, true_values=target_batch)

            if display_confusion_matrix or save_results:
                # calculations for confusion matrix
                # print(f'prediction_batch: {prediction_batch}')
                # print(f'prediction_batch.shape: {prediction_batch.shape}')
                # print(f'target_batch: {target_batch}')
                # print(f'target_batch.shape: {target_batch.shape}')
                tag_list.extend(target_batch.tolist())
                prediction_batch = softmax(prediction_batch)
                # print(f'prediction_batch: {prediction_batch}')
                # print(f'prediction_batch.shape: {prediction_batch.shape}')
                top_class_batch = prediction_batch.argmax(dim=1)
                # print(f'top_class_batch: {top_class_batch}')
                # print(f'top_class_batch.shape: {top_class_batch.shape}')
                prediction_list.extend(top_class_batch.tolist())
                # exit()

            test_loss += loss.item()
            batch_counter += 1

    test_loss = test_loss / len(test_data)

    metric_evaluations = test_metric_manager.compute()
    if verbose >= 1:
        metric_utils.print_formatted_results(
            title='TEST RESULTS',
            loss=test_loss,
            metrics=metric_evaluations,
        )

    if save_results:
        # add test loss and metrics to meta_data file
        # also add confusion matrix
        assert save_path is not None
        model_utils.save_test_results(
            model=model,
            cm_true_values=tag_list,
            cm_predictions=prediction_list,
            save_path=save_path,
            test_loss=test_loss,
            metric_evaluations=metric_evaluations,
        )

    if display_confusion_matrix:
        # Plot the confusion matrix
        visualization_utils.display_cm(
            true_values=tag_list,
            predictions=prediction_list,
            class_information=class_information,
        )

    return test_loss, metric_evaluations
