import torch
import shutil
import random
from pathlib import Path

import global_constants as gc
from image_processing import resize
from data_preprocessing import data_loading, custom_dataset, balancing_augmentation


def get_data(data_path: str,
             batch_size: int,
             shuffle: bool,
             balance_data: bool,
             train_val_test_proportions: list,
             no_resizing: bool = False,
             single_dataloader: bool = False,
             standard_img_dim: int = None,
             custom_transforms: list = None,
             use_only_classes: list = None,
             model_class_information: dict = None,
             augmentation_proportion: int = 1,
             random_seed: int = None,
             verbose: int = 0,
             ):
    # TODO: this is a temporary solution. each time delete the step 2 data and compute them again. To load them,
    #  step 2 data must be divided in folders based on the custom_transforms applied, batch_size, shuffle, and so
    #  on. This is not implemented yet.
    augmentation_path = f'{gc.STEP_2_DATA_PATH}augmentation_{augmentation_proportion}/'
    shutil.rmtree(path=gc.STEP_2_DATA_PATH, ignore_errors=True)

    if verbose >= 1:
        print('Loading data...')

    img_list, tag_list, class_information = data_loading.load_data(
        data_path=data_path,
        use_only_classes=use_only_classes,
        model_class_information=model_class_information,
        verbose=verbose,
    )
    if not no_resizing:
        # resize images
        img_list = resize.resize_img_list(
            img_list=img_list,
            standard_img_dim=standard_img_dim,
            verbose=verbose,
        )

    img_original_pixel_size = resize.get_mean_pixel_size(img_list=img_list, verbose=verbose)

    # split dataset
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
            class_information=class_information,
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

    if single_dataloader:
        # combine train, val, and test data in a single dataloader
        train_val_test_img_list = train_val_test_img_list[0] + train_val_test_img_list[1] + train_val_test_img_list[2]
        train_val_test_tags_list = train_val_test_tags_list[0] + train_val_test_tags_list[1] + train_val_test_tags_list[2]
        # create dataset
        train_ds = custom_dataset.Dataset_from_obs_targets(
            obs_list=train_val_test_img_list,
            target_list=train_val_test_tags_list,
            # name='complete_dataset',
        )
        if verbose >= 1:
            print('Single single_dataset created')
            print(f'single_dataset length: {len(train_ds)}')
        # create data loader
        train_dl = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        if verbose >= 1:
            print('Single dataloader created')

    else:
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
    complete_file_path = f'{augmentation_path}{gc.DL_FILE_NAME}{gc.PYTORCH_FILE_EXTENSION}'
    augmentation_path = Path(augmentation_path)
    # if not augmentation_path.exists():
    #     augmentation_path.mkdir(parents=True)
    # torch.save(obj=(train_dl, val_dl, test_dl), f=complete_file_path)
    if verbose >= 1:
        print('Step 2 data generated')

    if single_dataloader:
        return train_dl, img_shape, img_original_pixel_size, class_information
    else:
        return train_dl, val_dl, test_dl, img_shape, img_original_pixel_size, class_information
