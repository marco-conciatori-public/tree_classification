import utils
import global_constants
from import_args import args
from models import evaluation, model_utils
from data_preprocessing import data_loading, get_ready_data
from biodiversity_metrics import gini_simpson_index, species_richness, shannon_wiener_index


def compute_biodiversity_(**kwargs):
    # import parameters
    parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH, **kwargs)
    # use_targets = parameters['use_targets']
    use_targets = True
    use_network = True
    assert use_targets, 'use_targets must be True for now'
    assert use_targets or use_network, 'At least one of the two parameters ("use_targets", "use_network") must be True'

    if use_network:
        partial_name = str(
            input('Insert name or part of the name of a model: '))
        model_id = int(input('Insert model id number: '))
        model_path, info_path = utils.get_path_by_id(
            partial_name=partial_name,
            model_id=model_id,
            folder_path=global_constants.MODEL_OUTPUT_DIR,
        )
        loaded_model, custom_transforms, meta_data = model_utils.load_model(
            model_path=model_path,
            device=parameters['device'],
            training_mode=False,
            meta_data_path=info_path,
            verbose=parameters['verbose'],
        )

        _, _, test_dl, _, _, _ = get_ready_data.get_data(
            data_path=parameters['data_path'],
            shuffle=True,
            balance_data=False,
            batch_size=parameters['batch_size'],
            train_val_test_proportions=[0.01, 0.01, 0.98],
            # standard_img_dim=config.IMG_DIM,
            no_resizing=True,
            custom_transforms=custom_transforms,
            use_only_classes=parameters['use_only_classes'],
            augmentation_proportion=1,
            random_seed=parameters['random_seed'],
            verbose=parameters['verbose'],
        )

        _, _ = evaluation.eval(
            model=loaded_model,
            test_data=test_dl,
            loss_function_name=parameters['loss_function_name'],
            device=parameters['device'],
            class_information=meta_data['class_information'],
            display_confusion_matrix=parameters['display_confusion_matrix'],
            metrics=parameters['metrics'],
            save_results=parameters['save_model'],
            save_path=global_constants.MODEL_OUTPUT_DIR,
            verbose=parameters['verbose'],
        )

    else:
        _, tag_list, class_information = data_loading.load_data(
            data_path=parameters['data_path'],
            use_targets=use_targets,
            use_only_classes=parameters['use_only_classes'],
            verbose=parameters['verbose'],
        )
        print(f'tag_list length: {len(tag_list)}')

        gs_index = gini_simpson_index.get_bio_diversity_index(
            tag_list=tag_list,
            class_information=class_information,
        )
        sw_index = shannon_wiener_index.get_bio_diversity_index(
            tag_list=tag_list,
            class_information=class_information,
        )
        sr_index = species_richness.get_bio_diversity_index(tag_list=tag_list)
        print(f'Gini-Simpson index: {gs_index}')
        print(f'Shannon-Wiener index: {sw_index}')
        print(f'Species richness: {sr_index}')


if __name__ == '__main__':
    compute_biodiversity_()
