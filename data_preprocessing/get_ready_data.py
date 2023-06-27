import torch
from pathlib import Path

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

    if verbose >= 1:
        print('Loading data...')
    try:
        train_dl, val_dl, test_dl = torch.load(
            f=global_constants.FINAL_DATA_PATH + global_constants.DL_FILE_NAME + global_constants.PYTORCH_FILE_EXTENSION
        )
        print('Data loader found and loaded.')
        batched_img_tag = next(iter(train_dl))
        batched_img_shape = batched_img_tag[0].shape
        # print(f'batched_img_shape: {batched_img_shape}.')
        # print(f'batched Target shape: {batched_img_tag[1].shape}.')
        # remove batch dimension
        img_shape = batched_img_shape[1:]
        # print(f'img_shape: {img_shape}.')
        return train_dl, val_dl, test_dl, img_shape
    except Exception:
        print('Data loader not found, generating one.')

    preprocessed_data_loaded = True
    try:
        img_list, tag_list = data_loading.load_data(
            data_path=global_constants.PREPROCESSED_DATA_PATH,
            verbose=verbose,
        )
        print('Preprocessed data found and loaded.')
    except Exception:
        print('Preprocessed data not found, generating them.')
        preprocessed_data_loaded = False

    if preprocessed_data_loaded:
        if len(img_list) == 0:
            print('Preprocessed data found but incorrect, re-generating them.')
            preprocessed_data_loaded = False

    if not preprocessed_data_loaded:
        img_path_list = data_loading.get_img_path_list(global_constants.INTERMEDIATE_DATA_PATH, verbose=verbose)
        # # get min width and height separately
        min_width, min_height = standardize_img.get_min_dimensions(img_path_list)
        if verbose >= 2:
            print(f'Minimum width: {min_width}, minimum height: {min_height}.')

        for img_path in img_path_list:
            # resize images to the smallest width and height found in the dataset
            # also save results in preprocessed data folder
            standardize_img.resize_img(
                img_path=img_path,
                min_width=min_width,
                min_height=min_height,
            )
        if verbose >= 1:
            print('Preprocessed data saved.')

        img_list, tag_list = data_loading.load_data(
            data_path=global_constants.PREPROCESSED_DATA_PATH,
            verbose=verbose,
        )

    # apply data augmentation
    temp_img_list = []
    temp_tag_list = []
    if augment_data > 1:
        if verbose >= 1:
            print(f'Applying data augmentation. Num original obs: {len(img_list)}.')
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
        if verbose >= 1:
            print(f'Data augmentation applied. Num obs after augmentation: {len(img_list)}.')

    utils.check_split_proportions(train_val_test_proportions=train_val_test_proportions, tolerance=tolerance)

    # create dataset
    ds = custom_dataset.Dataset_from_obs_targets(
        obs_list=img_list,
        target_list=tag_list,
        # name='complete_dataset',
    )
    # split dataset
    total_length = len(ds)
    split_lengths = [int(total_length * proportion) for proportion in train_val_test_proportions]
    split_lengths[2] = total_length - split_lengths[0] - split_lengths[1]
    train_ds, val_ds, test_ds = ds.random_split(lengths=split_lengths)

    # create data loaders
    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    val_dl = torch.utils.data.DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    test_dl = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # get image shape
    batched_img_tag = next(iter(train_dl))
    batched_img_shape = batched_img_tag[0].shape
    # print(f'batched_img_shape: {batched_img_shape}.')
    # print(f'batched Target shape: {batched_img_tag[1].shape}.')
    # remove batch dimension
    img_shape = batched_img_shape[1:]
    # print(f'img_shape: {img_shape}.')
    final_data_path = Path(global_constants.FINAL_DATA_PATH)
    if not final_data_path.exists():
        final_data_path.mkdir(parents=False)
    complete_file_path = global_constants.FINAL_DATA_PATH + global_constants.DL_FILE_NAME\
                         + global_constants.PYTORCH_FILE_EXTENSION
    torch.save(obj=(train_dl, val_dl, test_dl), f=complete_file_path)
    if verbose >= 1:
        print('Data Generated.')
        print('Data loader saved.')

    return train_dl, val_dl, test_dl, img_shape
