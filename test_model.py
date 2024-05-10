import torch

import utils
import global_constants
from import_args import args
from models import model_utils
from metrics import metric_utils
from visualization import visualization_utils
from data_preprocessing import get_ready_data
from metrics.metric_manager import MetricManager


def test_model_(**kwargs):
    # import parameters
    parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH, **kwargs)
    parameters['verbose'] = 2
    parameters['shuffle'] = False

    device = parameters['device']
    if parameters['verbose'] >= 1:
        print(f'data_path: {parameters["data_path"]}')

    partial_name, model_id = utils.identify_model(parameters=parameters)
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

    test_dl, _, _, _ = get_ready_data.get_data(
        data_path=parameters['data_path'],
        shuffle=parameters['shuffle'],
        balance_data=False,
        batch_size=1,
        train_val_test_proportions=parameters['train_val_test_proportions'],
        # standard_img_dim=config.IMG_DIM,
        single_dataloader=True,
        use_only_classes=parameters['use_only_classes'],
        model_class_information=meta_data['class_information'],
        custom_transforms=custom_transforms,
        augmentation_proportion=1,
        random_seed=parameters['random_seed'],
        verbose=parameters['verbose'],
    )
    if parameters['verbose'] >= 1:
        print(f'len(test_dl): {len(test_dl)}')

    # get loss function from string name
    loss_function = getattr(torch.nn, parameters['loss_function_name'])()

    test_metric_manager = MetricManager(
        biodiversity_metric_names=parameters['metric_names']['biodiversity'],
        classification_metric_names=parameters['metric_names']['classification'],
        class_information=meta_data['class_information'],
    )

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
            test_metric_manager.update(predicted_probabilities=prediction_batch, true_values=target_batch)

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

    if parameters['verbose'] >= 1:
        metric_utils.print_formatted_results(
            title='TEST RESULTS',
            loss=test_loss,
            metrics=test_metric_manager.compute(),
        )

    if parameters['display_confusion_matrix']:
        # Plot the confusion matrix
        visualization_utils.display_cm(
            true_values=tag_list,
            predictions=prediction_list,
            class_information=meta_data['class_information'],
        )


if __name__ == '__main__':
    # pair_input_list = [
    #     ('convnext', 1),
    #     ('regnet_y', 0),
    #     ('resnet50', 0),
    #     ('swin_t', 0),
    #     ('swin_t', 1),
    #     ('swin_t', 2),
    #     ('swin_t', 3),
    #     ('swin_t', 4),
    # ]
    # input_data_folder = 'step_1_test_5_species'
    display_confusion_matrix = True
    # for partial_name, model_id in pair_input_list:
    #     test_model_(
    #         partial_name=partial_name,
    #         model_id=model_id,
    #         input_data_folder=input_data_folder,
    #         display_confusion_matrix=display_confusion_matrix,
    #     )
    test_model_()
