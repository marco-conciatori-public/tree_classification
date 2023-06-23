import torch

import utils
import global_constants
from data_preprocessing import data_loading, standardize_img, custom_dataset, data_augmentation


def get_data(batch_size: int,
             shuffle: bool,
             train_val_test_proportions: list,
             tolerance: float,
             augment_data: int = 1,
             verbose: int = 0,
             ):
    data_loaded = True
    try:
        img_list, tag_list = data_loading.load_data(
            img_folder_path=global_constants.PREPROCESSED_DATA_PATH,
            verbose=verbose,
        )
    except:
        data_loaded = False

    if not data_loaded or len(img_list) == 0:
        img_path_list = data_loading.load_img(global_constants.INTERMEDIATE_DATA_PATH, verbose=verbose)
        # # get min width and height separately
        min_width, min_height = standardize_img.get_min_dimensions(img_path_list)
        if verbose >= 2:
            print(f'Minimum width: {min_width}, minimum height: {min_height}.')

        str_img_path_list = []
        for img_path in img_path_list:
            # resize images to the smallest width and height found in the dataset
            # also save results in preprocessed data folder
            standardize_img.resize_img(
                img_path=img_path,
                min_width=min_width,
                min_height=min_height,
            )
        print('Preprocessed data saved.')

        img_list, tag_list = data_loading.load_data(
            img_folder_path=global_constants.PREPROCESSED_DATA_PATH,
            verbose=verbose,
        )

    # apply data augmentation
    temp_img_list = []
    temp_tag_list = []
    if augment_data > 1:
        for i in range(augment_data - 1):
            new_img_list, new_tag_list = data_augmentation.random_transform_img_list(
                img_list=img_list,
                tag_list=tag_list,
                # apply_probability=0.6,
            )
            temp_img_list.extend(new_img_list)
            temp_tag_list.extend(new_tag_list)
    img_list.extend(temp_img_list)
    tag_list.extend(temp_tag_list)

    utils.check_split_proportions(train_val_test_proportions=train_val_test_proportions, tolerance=tolerance)

    # create dataset
    ds = custom_dataset.Dataset_from_obs_targets(
        obs_list=img_list,
        target_list=tag_list,
        # name='complete_dataset',
    )

    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return img_list, tag_list