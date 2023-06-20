import torch
import torchmetrics
import ignite.metrics

from models import model_utils


def eval(model: torch.nn.Module,
         test_data,
         loss_function_name: str,
         device: torch.device,
         max_loss_coefficient: int = None,
         meta_data: dict = None,
         metrics=(),
         verbose: int = 0,
         ):

    # get loss function from string name
    loss_function = getattr(torch.nn, loss_function_name)()

    test_metrics = {}
    for metric_name in metrics:
        try:
            metric_class = getattr(torchmetrics, metric_name)
        except AttributeError:
            try:
                metric_class = getattr(ignite.metrics, metric_name)
            except AttributeError:
                raise AttributeError(f'metric {metric_name} not found in custom metrics, torchmetrics or ignite.')

        test_metric_instance = metric_class()

        test_metrics[metric_name] = test_metric_instance

    test_loss = 0.0
    batch_counter = 0
    model.eval()
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

            # update metrics
            for metric in test_metrics.values():
                metric.update(prediction_batch.cpu(), target_batch.cpu())

            test_loss += loss.item()
            batch_counter += 1

    test_loss = test_loss / len(test_data)
    metric_evaluations = {}
    for metric_name in test_metrics:
        metric = test_metrics[metric_name]
        metric_evaluations[metric_name] = metric.compute()

    if verbose >= 1:
        model_utils.print_formatted_results(
            title='TEST RESULTS',
            loss=test_loss,
            metrics=metric_evaluations,
        )

    return test_loss, metric_evaluations
