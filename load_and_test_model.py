import torch
import torchmetrics

import utils
import config
import global_constants
from models import model_utils
from data_preprocessing import get_ready_data


# PARAMETERS
verbose = 2
model_id = 0
partial_name = 'regnety'
show_confusion_matrix = True
device = utils.get_available_device(verbose=verbose)

if show_confusion_matrix:
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

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
    verbose=verbose,
)

_, _, test_dl, img_shape = get_ready_data.get_data(
    shuffle=config.SHUFFLE,
    balance_data=False,
    batch_size=1,
    train_val_test_proportions=config.TRAIN_VAL_TEST_PROPORTIONS,
    # standard_img_dim=config.IMG_DIM,
    custom_transforms=custom_transforms,
    tolerance=config.TOLERANCE,
    augmentation_proportion=1,
    verbose=verbose,
)
if verbose >= 1:
    print(f'len(test_dl): {len(test_dl)}')

# get loss function from string name
loss_function = getattr(torch.nn, config.LOSS_FUNCTION_NAME)()

test_metrics = {}
for metric_name, metric_args in config.METRICS.items():
    try:
        metric_class = getattr(torchmetrics, metric_name)
    except AttributeError:
        raise AttributeError(f'metric {metric_name} not found in torchmetrics')

    test_metrics[metric_name] = metric_class(**metric_args)

softmax = torch.nn.Softmax(dim=0)
top_predictions = []
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
        top_predictions.append(top_class)
        tag_list.append(target_batch.squeeze(0).item())

test_loss = test_loss / len(test_dl)
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

if show_confusion_matrix:
    # Plot the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_true=tag_list,
        y_pred=top_predictions,
        display_labels=global_constants.TREE_CATEGORIES_JAPANESE,
        xticks_rotation=45,
    )
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()
