import global_constants
from import_args import args
from data_preprocessing import get_ready_data
from models import model_utils, training, evaluation


def train_custom_model_(**kwargs):
    # import parameters
    parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH, **kwargs)

    train_dl, val_dl, test_dl, img_shape = get_ready_data.get_data(
        data_path=parameters['data_path'],
        batch_size=parameters['batch_size'],
        shuffle=parameters['shuffle'],
        balance_data=parameters['balance_data'],
        train_val_test_proportions=parameters['train_val_test_proportions'],
        # standard_img_dim=parameters['img_dim'],
        augmentation_proportion=parameters['data_augmentation_proportion'],
        random_seed=parameters['random_seed'],
        verbose=parameters['verbose'],
    )

    model = model_utils.create_model(
        model_class_name='Conv_2d',
        input_shape=img_shape,
        num_output=len(global_constants.TREE_INFORMATION),
        custom_model_parameters=parameters['custom_model_parameters'],
        device=parameters['device'],
        name='test_conv_2d',
        verbose=parameters['verbose'],
    )
    # print(model)
    # temp_tensor = torch.Tensor(img_list[0])
    # # add batch dimension
    # print(f'main tensor shape: {temp_tensor.shape}')
    # temp_tensor = temp_tensor.unsqueeze(0)
    # print(f'main tensor shape: {temp_tensor.shape}')
    # # switch from HWC to CHW
    # temp_tensor = temp_tensor.permute(0, 3, 1, 2)
    # print(f'main tensor shape: {temp_tensor.shape}')
    # temp_tensor = temp_tensor.to(parameters['device'])
    # print(f'main tensor shape: {temp_tensor.shape}')
    # result = model(temp_tensor)
    # print(result)

    parameters_to_save = {}
    parameters_to_save['shuffle'] = parameters['shuffle']
    parameters_to_save['random_seed'] = parameters['random_seed']
    parameters_to_save['augmentation_proportion'] = parameters['data_augmentation_proportion']
    parameters_to_save['balance_classes'] = parameters['balance_data']
    parameters_to_save['custom_model_parameters'] = parameters['balance_data']
    training_history = training.train(
        model=model,
        training_data=train_dl,
        validation_data=val_dl,
        num_epochs=parameters['num_epochs'],
        learning_rate=parameters['learning_rate'],
        loss_function_name=parameters['loss_function_name'],
        optimizer_name=parameters['optimizer_name'],
        device=parameters['device'],
        verbose=parameters['verbose'],
        save_model=parameters['save_model'],
        save_path=global_constants.MODEL_OUTPUT_DIR,
        metrics=parameters['metrics'],
        extra_info_to_save=parameters_to_save,
    )
    print(f'training_history:\n{training_history}')

    test_loss, metric_evaluations = evaluation.eval(
        model=model,
        test_data=test_dl,
        loss_function_name=parameters['loss_function_name'],
        device=parameters['device'],
        display_confusion_matrix=parameters['display_confusion_matrix'],
        metrics=parameters['metrics'],
        save_results=parameters['save_model'],
        save_path=global_constants.MODEL_OUTPUT_DIR,
        notebook_mode=parameters['notebook_mode'],
        verbose=parameters['verbose'],
    )
    print(f'test_loss: {test_loss}')
    print(f'test_metric_evaluations: {metric_evaluations}')


if __name__ == '__main__':
    train_custom_model_()
