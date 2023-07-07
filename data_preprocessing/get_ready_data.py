import torch
from pathlib import Path

import utils
import global_constants
from data_preprocessing import data_loading, standardize_img, custom_dataset, data_augmentation


def get_data(batch_size: int,
             shuffle: bool,
             train_val_test_proportions: list,
             tolerance: float,
             custom_transforms: list = None,
             augment_data: int = 1,
             verbose: int = 0,
             ):

    if verbose >= 1:
        print('Loading data...')
    try:
        train_dl, val_dl, test_dl = torch.load(
            f=global_constants.STEP_3_DATA_PATH + global_constants.DL_FILE_NAME + global_constants.PYTORCH_FILE_EXTENSION
        )
        print('Data loader found and loaded')
        batched_img_tag = next(iter(train_dl))
        batched_img_shape = batched_img_tag[0].shape
        # print(f'batched_img_shape: {batched_img_shape}')
        # print(f'batched Target shape: {batched_img_tag[1].shape}')
        # remove batch dimension
        img_shape = batched_img_shape[1:]
        # print(f'img_shape: {img_shape}')
        return train_dl, val_dl, test_dl, img_shape
    except Exception:
        print('Data loader not found, generating one')

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
        # # get min width and height separately
        min_width, min_height = standardize_img.get_min_dimensions(img_path_list)
        if verbose >= 2:
            print(f'Minimum width: {min_width}, minimum height: {min_height}')

        for img_path in img_path_list:
            # resize images to the smallest width and height found in the dataset
            # also save results in step_2_data folder
            standardize_img.resize_img(
                img_path=img_path,
                min_width=min_width,
                min_height=min_height,
            )
        if verbose >= 1:
            print('Step 2 data saved')

        img_list, tag_list = data_loading.load_data(
            data_path=global_constants.STEP_2_DATA_PATH,
            verbose=verbose,
        )

    # apply custom transforms
    temp_img_list = []
    for img in img_list:
        for transform in custom_transforms:
            img = transform(img)

        if torch.is_tensor(img):
            img = img.numpy()
        temp_img_list.append(img)
    img_list = temp_img_list

    img_list, tag_list = data_augmentation.apply_data_augmentation(
        img_list=img_list,
        tag_list=tag_list,
        augment_data=augment_data,
        verbose=verbose,
    )

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
    step_3_data_path = Path(global_constants.STEP_3_DATA_PATH)
    if not step_3_data_path.exists():
        step_3_data_path.mkdir(parents=False)
    complete_file_path = global_constants.STEP_3_DATA_PATH + global_constants.DL_FILE_NAME \
                         + global_constants.PYTORCH_FILE_EXTENSION
    torch.save(obj=(train_dl, val_dl, test_dl), f=complete_file_path)
    if verbose >= 1:
        print('Step 3 data generated')
        print('Step 3 data saved')

    return train_dl, val_dl, test_dl, img_shape
