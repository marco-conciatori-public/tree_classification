import utils
import global_constants
from import_args import args
from data_preprocessing import get_ready_data
from models import training, evaluation, model_utils


def finetune_many_models_separate_test_(**kwargs):
    # import parameters
    parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH, **kwargs)
    parameters['display_confusion_matrix'] = True
    if parameters['num_models_to_train'] > 1:
        parameters['display_confusion_matrix'] = False

    average_by_metric = {}
    for model_num in range(parameters['num_models_to_train']):
        print(f'Model {model_num + 1} of {parameters["num_models_to_train"]}')

        # get data
        custom_transforms, resize_in_attributes = model_utils.get_custom_transforms(
            weights_name=parameters['pretrained_model_parameters']['weights_name'],
            verbose=parameters['verbose'],
        )
        train_dl, val_dl, _, img_shape, img_original_pixel_size, class_information = get_ready_data.get_data(
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

        if parameters['verbose'] >= 2:
            # check image shape
            print(f'img_shape: {img_shape}')

            batched_img_tag = next(iter(train_dl))
            batched_img_shape = batched_img_tag[0].shape
            print(f'batched_img_shape: {batched_img_shape}')
            print(f'batched target shape: {batched_img_tag[1].shape}')
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
        _ = training.train(
            model=model,
            training_data=train_dl,
            validation_data=val_dl,
            num_epochs=parameters['num_epochs'],
            loss_function_name=parameters['loss_function_name'],
            optimizer_parameters=parameters['optimizer_parameters'],
            device=parameters['device'],
            class_information=class_information,
            verbose=0,
            save_model=parameters['save_model'],
            save_path=global_constants.MODEL_OUTPUT_DIR,
            metrics=parameters['metric_names'],
            custom_transforms=custom_transforms,
            extra_info_to_save=parameters_to_save,
        )

        if isinstance(parameters['test_data_path'], list):
            assert len(parameters['test_data_path']) == parameters['num_models_to_train'], \
                'The number of test data paths must be equal to the number of models to train, if it is a list.'
            test_data_path_ = parameters['test_data_path'][model_num]

        else:
            test_data_path_ = parameters['test_data_path']
        test_dl, _, _, class_information = get_ready_data.get_data(
            data_path=test_data_path_,
            batch_size=parameters['batch_size'],
            shuffle=parameters['shuffle'],
            balance_data=parameters['balance_data'],
            custom_transforms=custom_transforms,
            train_val_test_proportions=parameters['train_val_test_proportions'],
            no_resizing=resize_in_attributes,
            use_only_classes=parameters['use_only_classes'],
            augmentation_proportion=1,
            single_dataloader=True,
            random_seed=parameters['random_seed'],
            verbose=parameters['verbose'],
        )
        num_classes_test = len(class_information)
        test_loss, metric_evaluations = evaluation.eval(
            model=model,
            test_data=test_dl,
            loss_function_name=parameters['loss_function_name'],
            device=parameters['device'],
            display_confusion_matrix=parameters['display_confusion_matrix'],
            metrics=parameters['metric_names'],
            class_information=class_information,
            save_results=parameters['save_model'],
            save_path=global_constants.MODEL_OUTPUT_DIR,
            verbose=parameters['verbose'],
        )
        # print(f'test_loss: {test_loss}')
        for metric in metric_evaluations:
            if metric not in average_by_metric:
                average_by_metric[metric] = {}
            for species_id in range(num_classes_test):
                species_name = class_information[species_id][global_constants.SPECIES_LANGUAGE]
                if species_name not in average_by_metric[metric]:
                    average_by_metric[metric][species_name] = 0
                average_by_metric[metric][species_name] += metric_evaluations[metric]['result'][species_id]
        if 'loss' not in average_by_metric:
            average_by_metric['loss'] = 0
        average_by_metric['loss'] += test_loss

    for metric in average_by_metric:
        if metric == 'loss':
            average_by_metric[metric] /= parameters['num_models_to_train']
            continue
        total_all_species = 0
        for species_name in average_by_metric[metric]:
            average_by_metric[metric][species_name] /= parameters['num_models_to_train']
            total_all_species += average_by_metric[metric][species_name]
        average_by_metric[metric]['all_species'] = total_all_species / num_classes_test
    print('average_by_metric:')
    utils.pretty_print_dict(average_by_metric)


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
    # train_val_test_proportions = [0.09, 0.455, 0.455]
    # train_val_test_proportions = [0.01, 0.495, 0.495]
    # step = 0.01
    # for i in range(9):
    #     new_train_val_test_proportions = []
    #     new_train_val_test_proportions.append(round(train_val_test_proportions[0] - i * step, 3))
    #     new_train_val_test_proportions.append(round(train_val_test_proportions[1] + i * step / 2, 3))
    #     new_train_val_test_proportions.append(round(train_val_test_proportions[2] + i * step / 2, 3))
    #     print(f'new_train_val_test_proportions: {new_train_val_test_proportions}')
    #     finetune_pretrained_model_(train_val_test_proportions=new_train_val_test_proportions)

    # input_data_folder = 'step_1_less_data/balanced_'
    # for i in range(65, 0, -5):
    #     new_input_data_folder = f'{input_data_folder}{i}/'
    #     print(f'new_input_data_folder: {new_input_data_folder}')
    #     finetune_pretrained_model_(input_data_folder=new_input_data_folder, verbose=1)
    # either a single path for all the models or a list of paths, exactly one for model
    test_data_path = 'data/step_1_test_5_species/'
    finetune_many_models_separate_test_(test_data_path=test_data_path)
