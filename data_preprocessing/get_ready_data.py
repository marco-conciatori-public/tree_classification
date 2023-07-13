import torch
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

import utils
import global_constants
from data_preprocessing import data_loading, standardize_img, custom_dataset, balancing_augmentation


def get_data(batch_size: int,
             shuffle: bool,
             balance_data: bool,
             train_val_test_proportions: list,
             tolerance: float,
             standard_img_dim: tuple = None,
             custom_transforms: list = None,
             augmentation_proportion: int = 1,
             random_seed: int = None,
             verbose: int = 0,
             ):

    # TODO: dont resize images in step 2 if fine tuning
    # TODO: separate resizing from saving in step 2
    # TODO: maybe dont save step 2 data
    # TODO: this is a temporary solution. each time delete the step 3 data and compute them again.
    # To load them, step 3 data must be divided in folders based on the custom_transforms applied, batch_size,
    # shuffle, and so on. This is not implemented yet.
    augmentation_path = f'{global_constants.STEP_3_DATA_PATH}augmentation_{augmentation_proportion}/'
    shutil.rmtree(path=global_constants.STEP_3_DATA_PATH, ignore_errors=True)

    if verbose >= 1:
        print('Loading data...')
    # try:
    #     train_dl, val_dl, test_dl = torch.load(
    #         f=augmentation_path + global_constants.DL_FILE_NAME + global_constants.PYTORCH_FILE_EXTENSION
    #     )
    #     print('Step 3 data found and loaded')
    #     batched_img_tag = next(iter(train_dl))
    #     batched_img_shape = batched_img_tag[0].shape
    #     # print(f'batched_img_shape: {batched_img_shape}')
    #     # print(f'batched Target shape: {batched_img_tag[1].shape}')
    #     # remove batch dimension
    #     img_shape = batched_img_shape[1:]
    #     # print(f'img_shape: {img_shape}')
    #     return train_dl, val_dl, test_dl, img_shape
    # except Exception:
    #     print('Step 3 data not found, generating them')
    #
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

    # step 2
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

    # step 3
    # split dataset
    utils.check_split_proportions(train_val_test_proportions=train_val_test_proportions, tolerance=tolerance)
    total_length = len(img_list)
    split_lengths = [int(total_length * proportion) for proportion in train_val_test_proportions]
    split_lengths[2] = total_length - split_lengths[0] - split_lengths[1]
    if shuffle:
        # extract training data randomly
        train_imgs, val_and_test_imgs, train_tags, val_and_test_tags = train_test_split(
            img_list,
            tag_list,
            train_size=split_lengths[0],
            random_state=random_seed,
        )
        # extract validation and test data randomly
        val_imgs, test_imgs, val_tags, test_tags = train_test_split(
            val_and_test_imgs,
            val_and_test_tags,
            train_size=split_lengths[1],
            random_state=random_seed,
        )
    else:  # not shuffle
        train_imgs = img_list[ : split_lengths[0]]
        train_tags = tag_list[ : split_lengths[0]]
        val_imgs = img_list[split_lengths[0] : split_lengths[0] + split_lengths[1]]
        val_tags = tag_list[split_lengths[0] : split_lengths[0] + split_lengths[1]]
        test_imgs = img_list[split_lengths[0] + split_lengths[1] : ]
        test_tags = tag_list[split_lengths[0] + split_lengths[1] : ]

    if verbose >= 2:
        print('image list split')
        # print(f'split_lengths: {split_lengths}')
        print(f'train_imgs length: {len(train_imgs)}')
        print(f'val_imgs length: {len(val_imgs)}')
        print(f'test_imgs length: {len(test_imgs)}')
    assert len(train_imgs) == len(train_tags)
    assert len(val_imgs) == len(val_tags)
    assert len(test_imgs) == len(test_tags)
    
    if balance_data or (augmentation_proportion > 1):
        # apply data balancing/augmentation
        train_imgs, train_tags = balancing_augmentation.balance_augment_data(
            img_list=train_imgs,
            tag_list=train_tags,
            balance_data=balance_data,
            augmentation_proportion=augmentation_proportion,
            verbose=verbose,
        )
    
    train_val_test_img_list = [train_imgs, val_imgs, test_imgs]
    train_val_test_tags_list = [train_tags, val_tags, test_tags]
    
    # apply custom transforms
    if custom_transforms is not None:
        for index in range(len(train_val_test_img_list)):
            temp_img_list = []
            for img in train_val_test_img_list[index]:
                for transform in custom_transforms:
                    img = transform(img)
    
                temp_img_list.append(img)
            train_val_test_img_list[index] = temp_img_list
    
    # create datasets
    train_val_test_ds = []
    for index in range(len(train_val_test_img_list)):
        train_val_test_ds.append(
            custom_dataset.Dataset_from_obs_targets(
                obs_list=train_val_test_img_list[index],
                target_list=train_val_test_tags_list[index],
                # name='complete_dataset',
            )
        )

    if verbose >= 1:
        print('Datasets created')
        if verbose >= 2:
            print(f'train_val_test_ds[0] length: {len(train_val_test_ds[0])}')
            print(f'train_val_test_ds[1] length: {len(train_val_test_ds[1])}')
            print(f'train_val_test_ds[2] length: {len(train_val_test_ds[2])}')

    # create data loaders
    train_dl = torch.utils.data.DataLoader(
        dataset=train_val_test_ds[0],
        batch_size=batch_size,
        shuffle=shuffle,
    )
    val_dl = torch.utils.data.DataLoader(
        dataset=train_val_test_ds[1],
        batch_size=batch_size,
        shuffle=shuffle,
    )
    test_dl = torch.utils.data.DataLoader(
        dataset=train_val_test_ds[2],
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
