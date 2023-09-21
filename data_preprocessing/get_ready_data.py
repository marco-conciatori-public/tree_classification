import torch
import shutil
import random
from pathlib import Path

import utils
import global_constants
from image_processing import resize
from data_preprocessing import data_loading, custom_dataset, balancing_augmentation


def get_data(batch_size: int,
             shuffle: bool,
             balance_data: bool,
             train_val_test_proportions: list,
             tolerance: float,
             no_resizing: bool = False,
             standard_img_dim: int = None,
             custom_transforms: list = None,
             augmentation_proportion: int = 1,
             random_seed: int = None,
             verbose: int = 0,
             ):
    # TODO: this is a temporary solution. each time delete the step 2 data and compute them again. To load them,
    #  step 2 data must be divided in folders based on the custom_transforms applied, batch_size, shuffle, and so
    #  on. This is not implemented yet.
    augmentation_path = f'{global_constants.STEP_2_DATA_PATH}augmentation_{augmentation_proportion}/'
    shutil.rmtree(path=global_constants.STEP_2_DATA_PATH, ignore_errors=True)

    if verbose >= 1:
        print('Loading data...')

    img_list, tag_list = data_loading.load_data(
        data_path=global_constants.STEP_1_DATA_PATH,
        verbose=verbose,
    )
    if not no_resizing:
        # resize images
        img_list = resize.resize_img_list(
            img_list=img_list,
            standard_img_dim=standard_img_dim,
            verbose=verbose,
        )

    # split dataset
    utils.check_split_proportions(train_val_test_proportions=train_val_test_proportions, tolerance=tolerance)
    total_length = len(img_list)
    split_lengths = [int(total_length * proportion) for proportion in train_val_test_proportions]
    split_lengths[2] = total_length - split_lengths[0] - split_lengths[1]
    indices = list(range(total_length))
    if shuffle:
        train_imgs = []
        train_tags = []
        val_imgs = []
        val_tags = []
        test_imgs = []
        test_tags = []
        random.shuffle(indices)
        # extract training, validation, and test data randomly
        for index in indices:
            if len(train_imgs) < split_lengths[0]:
                train_imgs.append(img_list[index])
                train_tags.append(tag_list[index])
            elif len(val_imgs) < split_lengths[1]:
                val_imgs.append(img_list[index])
                val_tags.append(tag_list[index])
            else:
                test_imgs.append(img_list[index])
                test_tags.append(tag_list[index])

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
        print('Step 2 data generated and saved')

    return train_dl, val_dl, test_dl, img_shape
