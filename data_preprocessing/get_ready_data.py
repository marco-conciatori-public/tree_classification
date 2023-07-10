import torch
from pathlib import Path

import utils
import global_constants
from data_preprocessing import data_loading, standardize_img, custom_dataset, data_augmentation


def get_data(batch_size: int,
             shuffle: bool,
             train_val_test_proportions: list,
             tolerance: float,
             standard_img_dim: tuple = None,
             custom_transforms: list = None,
             augment_data: int = 1,
             verbose: int = 0,
             ):

    if verbose >= 1:
        print('Loading data...')
    augmentation_path = f'{global_constants.STEP_3_DATA_PATH}augmentation_{augment_data}/'
    try:
        train_dl, val_dl, test_dl = torch.load(
            f=augmentation_path + global_constants.DL_FILE_NAME + global_constants.PYTORCH_FILE_EXTENSION
        )
        print('Step 3 data found and loaded')
        batched_img_tag = next(iter(train_dl))
        batched_img_shape = batched_img_tag[0].shape
        # print(f'batched_img_shape: {batched_img_shape}')
        # print(f'batched Target shape: {batched_img_tag[1].shape}')
        # remove batch dimension
        img_shape = batched_img_shape[1:]
        # print(f'img_shape: {img_shape}')
        return train_dl, val_dl, test_dl, img_shape
    except Exception:
        print('Step 3 data not found, generating them')

    step_2_data_loaded = True
    try:
        img_list, tag_list = data_loading.load_data(
            data_path=global_constants.STEP_2_DATA_PATH,
            verbose=verbose,
        )
        print('Step 2 data found and loaded')
    except Exception:
        print('Step 2 data not found, generating them')
        step_2_data_loaded = False

    if step_2_data_loaded:
        if len(img_list) == 0:
            print('Step 2 data found but incorrect, re-generating them')
            step_2_data_loaded = False

    if not step_2_data_loaded:
        img_path_list = data_loading.get_img_path_list(global_constants.STEP_1_DATA_PATH, verbose=verbose)
        if standard_img_dim is None:
            # # get min width and height separately
            width, height = standardize_img.get_min_dimensions(img_path_list)
            if verbose >= 2:
                print(f'Minimum width: {width}, minimum height: {height}')
        else:
            width, height = standard_img_dim

        for img_path in img_path_list:
            # resize images to the width and height
            # also save results in step_2 folder
            standardize_img.resize_img(
                img_path=img_path,
                min_width=width,
                min_height=height,
            )
        if verbose >= 1:
            print('Step 2 data saved')

        img_list, tag_list = data_loading.load_data(
            data_path=global_constants.STEP_2_DATA_PATH,
            verbose=verbose,
        )

    img_list, tag_list = data_augmentation.apply_data_augmentation(
        img_list=img_list,
        tag_list=tag_list,
        augment_data=augment_data,
        verbose=verbose,
    )

    # apply custom transforms
    temp_img_list = []
    for img in img_list:
        for transform in custom_transforms:
            img = transform(img)

        temp_img_list.append(img)
    img_list = temp_img_list
    utils.check_split_proportions(train_val_test_proportions=train_val_test_proportions, tolerance=tolerance)

    # create dataset
    ds = custom_dataset.Dataset_from_obs_targets(
        obs_list=img_list,
        target_list=tag_list,
        # name='complete_dataset',
    )
    if verbose >= 1:
        print('Dataset created')
    # split dataset
    total_length = len(ds)
    split_lengths = [int(total_length * proportion) for proportion in train_val_test_proportions]
    split_lengths[2] = total_length - split_lengths[0] - split_lengths[1]
    train_ds, val_ds, test_ds = ds.random_split(lengths=split_lengths)
    if verbose >= 2:
        print('Dataset split')
        print(f'train_ds length: {len(train_ds)}')
        print(f'val_ds length: {len(val_ds)}')
        print(f'test_ds length: {len(test_ds)}')

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
    if verbose >= 1:
        print('Data loaders created')

    # get image shape
    batched_img_tag = next(iter(train_dl))
    batched_img_shape = batched_img_tag[0].shape
    # print(f'batched_img_shape: {batched_img_shape}')
    # print(f'batched Target shape: {batched_img_tag[1].shape}')
    # remove batch dimension
    img_shape = batched_img_shape[1:]
    # print(f'img_shape: {img_shape}')
    complete_file_path = f'{augmentation_path}{global_constants.DL_FILE_NAME}{global_constants.PYTORCH_FILE_EXTENSION}'
    augmentation_path = Path(augmentation_path)
    if not augmentation_path.exists():
        augmentation_path.mkdir(parents=True)
    torch.save(obj=(train_dl, val_dl, test_dl), f=complete_file_path)
    if verbose >= 1:
        print('Step 3 data generated and saved')

    return train_dl, val_dl, test_dl, img_shape
