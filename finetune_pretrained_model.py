import global_constants
from import_args import args
from data_preprocessing import get_ready_data
from models import training, evaluation, model_utils


def finetune_pretrained_model_(**kwargs):
    # import parameters
    parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH, **kwargs)
    parameters['display_confusion_matrix'] = True
    if parameters['num_models_to_train'] > 1:
        parameters['display_confusion_matrix'] = False

    for model_num in range(parameters['num_models_to_train']):
        print(f'Model {model_num + 1} of {parameters["num_models_to_train"]}')

        # get data
        custom_transforms, resize_in_attributes = model_utils.get_custom_transforms(
            weights_name=parameters['pretrained_model_parameters']['weights_name'],
            verbose=parameters['verbose'],
        )
        train_dl, val_dl, test_dl, img_shape, img_original_pixel_size, class_information = get_ready_data.get_data(
            data_path=parameters['data_path'],
            batch_size=parameters['batch_size'],
            shuffle=parameters['shuffle'],
            balance_data=parameters['balance_data'],
            custom_transforms=custom_transforms,
            train_val_test_proportions=parameters['train_val_test_proportions'],
            no_resizing=resize_in_attributes,
            use_only_classes=parameters['use_only_classes'],
            augmentation_proportion=parameters['data_augmentation_proportion'],
            random_seed=parameters['random_seed'],
            verbose=parameters['verbose'],
        )

        # load model
        model = model_utils.get_torchvision_model(
            pretrained_model_parameters=parameters['pretrained_model_parameters'],
            device=parameters['device'],
            training=True,
            num_classes=len(class_information),
            verbose=parameters['verbose'],
        )
        # print(f'model:\n{model}')

        # check image shape
        print(f'img_shape: {img_shape}')

        batched_img_tag = next(iter(train_dl))
        batched_img_shape = batched_img_tag[0].shape
        print(f'batched_img_shape: {batched_img_shape}')
        print(f'batched Target shape: {batched_img_tag[1].shape}')
        # remove batch dimension
        img_shape = batched_img_shape[1:]
        print(f'img_shape: {img_shape}')

        parameters_to_save = {}
        parameters_to_save['shuffle'] = parameters['shuffle']
        parameters_to_save['random_seed'] = parameters['random_seed']
        parameters_to_save['augmentation_proportion'] = parameters['data_augmentation_proportion']
        parameters_to_save['balance_classes'] = parameters['balance_data']
        parameters_to_save['img_original_pixel_size'] = img_original_pixel_size
        parameters_to_save['img_shape'] = img_shape
        parameters_to_save['class_information'] = class_information
        training_history = training.train(
            model=model,
            training_data=train_dl,
            validation_data=val_dl,
            num_epochs=parameters['num_epochs'],
            loss_function_name=parameters['loss_function_name'],
            optimizer_parameters=parameters['optimizer_parameters'],
            device=parameters['device'],
            class_information=class_information,
            verbose=parameters['verbose'],
            save_model=parameters['save_model'],
            save_path=global_constants.MODEL_OUTPUT_DIR,
            metrics=parameters['metrics'],
            custom_transforms=custom_transforms,
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
            class_information=class_information,
            save_results=parameters['save_model'],
            save_path=global_constants.MODEL_OUTPUT_DIR,
            verbose=parameters['verbose'],
        )
        print(f'test_loss: {test_loss}')
        print(f'test_metric_evaluations: {metric_evaluations}')


if __name__ == '__main__':
    # pretrained_model_parameters_0 = {
    #     'model_architecture': 'regnet',
    #     'model_version': 'regnet_y_1_6gf',
    #     'weights_name': 'RegNet_Y_1_6GF_Weights.DEFAULT',
    #     'freeze_layers': False,
    # }
    # pretrained_model_parameters_1 = {
    #     'model_architecture': 'swin_transformer',
    #     'model_version': 'swin_t',
    #     'weights_name': 'Swin_T_Weights.DEFAULT',
    #     'freeze_layers': False,
    # }
    # pretrained_model_parameters_list = [pretrained_model_parameters_0, pretrained_model_parameters_1]
    # for i in range(2):
    #     finetune_pretrained_model_(pretrained_model_parameters=pretrained_model_parameters_list[i])
    finetune_pretrained_model_()
