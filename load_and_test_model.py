import torch
import torchmetrics

import utils
import global_constants
import visualization.visualization_utils
from import_args import args
from models import model_utils
from data_preprocessing import get_ready_data


# import parameters
parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH)
parameters['verbose'] = 2
model_id = int(input('Insert model id number: '))
partial_name = str(input('Insert name or part of the name to distinguish between models with the same id number: '))

device = parameters['device']

model_path, info_path = utils.get_path_by_id(
    partial_name=partial_name,
    model_id=model_id,
    folder_path=global_constants.MODEL_OUTPUT_DIR,
)

loaded_model, custom_transforms, meta_data = model_utils.load_model(
    model_path=model_path,
    device=device,
    training_mode=False,
    meta_data_path=info_path,
    verbose=parameters['verbose'],
)

_, _, test_dl, img_shape = get_ready_data.get_data(
    data_path=parameters['data_path'],
    shuffle=parameters['shuffle'],
    balance_data=False,
    batch_size=1,
    train_val_test_proportions=parameters['train_val_test_proportions'],
    # standard_img_dim=config.IMG_DIM,
    custom_transforms=custom_transforms,
    augmentation_proportion=1,
    random_seed=parameters['random_seed'],
    verbose=parameters['verbose'],
)
if parameters['verbose'] >= 1:
    print(f'len(test_dl): {len(test_dl)}')

# get loss function from string name
loss_function = getattr(torch.nn, parameters['loss_function_name'])()

test_metrics = {}
for metric_name, metric_args in parameters['metrics'].items():
    try:
        metric_class = getattr(torchmetrics, metric_name)
    except AttributeError:
        raise AttributeError(f'metric {metric_name} not found in torchmetrics')

    test_metrics[metric_name] = metric_class(**metric_args)

softmax = torch.nn.Softmax(dim=0)
prediction_list = []
tag_list = []
test_loss = 0.0
loaded_model.eval()
with torch.set_grad_enabled(False):
    for batch in test_dl:
        observation_batch, target_batch = batch

        # Potentially transfer batch to GPU
        if observation_batch.device != device:
            observation_batch = observation_batch.to(device)
            target_batch = target_batch.to(device)

        prediction_batch = loaded_model(observation_batch)
        loss = loss_function(prediction_batch, target_batch)
        prediction_batch = prediction_batch.cpu()
        target_batch = target_batch.cpu()

        # update metrics
        for metric in test_metrics.values():
            metric.update(prediction_batch, target_batch)

        test_loss += loss.item()

        if parameters['display_confusion_matrix']:
            # calculations for confusion matrix
            prediction = prediction_batch.squeeze(0)
            # print(f'prediction shape: {prediction.shape}')
            # print(f'prediction: {prediction}')
            prediction = softmax(prediction)
            # print(f'prediction shape: {prediction.shape}')
            # print(f'prediction: {prediction}')
            prediction = prediction.numpy()
            # print(f'prediction shape: {prediction.shape}')
            # print(f'prediction: {prediction}')
            top_class = prediction.argmax()
            prediction_list.append(top_class)
            tag_list.append(target_batch.squeeze(0).item())

test_loss = test_loss / len(test_dl)
metric_evaluations = {}
for metric_name in test_metrics:
    metric = test_metrics[metric_name]
    metric_evaluations[metric_name] = metric.compute()

if parameters['verbose'] >= 1:
    model_utils.print_formatted_results(
        title='TEST RESULTS',
        loss=test_loss,
        metrics=metric_evaluations,
        metrics_in_percentage=True,
    )

if parameters['display_confusion_matrix']:
    # Plot the confusion matrix
    visualization.visualization_utils.display_cm(true_values=tag_list, predictions=prediction_list)
