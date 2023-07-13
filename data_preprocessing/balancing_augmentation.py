import cv2
import copy
import random
import warnings
import albumentations as ab
from albumentations.augmentations import transforms as ab_transforms

import global_constants


def random_transform_img(img, apply_probability: float = 0.5):
    temp_img = copy.deepcopy(img)
    # not clear if this is necessary
    # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    if random.uniform(0, 1) < apply_probability:
        temp_img = cv2.flip(temp_img, 1)
    if random.uniform(0, 1) < apply_probability:
        temp_img = cv2.flip(temp_img, 0)
    if random.uniform(0, 1) < apply_probability:
        temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_CLOCKWISE)
    if random.uniform(0, 1) < apply_probability:
        temp_img = cv2.rotate(temp_img, cv2.ROTATE_180)
    if random.uniform(0, 1) < apply_probability:
        temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    partial_transform = ab.Compose([
        ab_transforms.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=apply_probability,
        ),
        ab_transforms.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=apply_probability,
        ),
    ])
    temp_img = partial_transform(image=temp_img)['image']
    return temp_img


def random_transform_img_list(img_list: list, tag_list: list, apply_probability: float = 0.5) -> (list, list):
    transformed_img_list = []
    corresponding_tag_list = []
    partial_transform = ab.Compose([
        ab_transforms.HueSaturationValue(
            hue_shift_limit=3,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=apply_probability,
        ),
        ab_transforms.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=apply_probability,
        ),
    ])

    for index in range(len(img_list)):
        img = img_list[index]
        temp_img = copy.deepcopy(img)
        # not clear if this is necessary
        # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        if random.uniform(0, 1) < apply_probability:
            temp_img = cv2.flip(temp_img, 1)
        if random.uniform(0, 1) < apply_probability:
            temp_img = cv2.flip(temp_img, 0)
        if random.uniform(0, 1) < apply_probability:
            temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_CLOCKWISE)
        if random.uniform(0, 1) < apply_probability:
            temp_img = cv2.rotate(temp_img, cv2.ROTATE_180)
        if random.uniform(0, 1) < apply_probability:
            temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        temp_img = partial_transform(image=temp_img)['image']
        transformed_img_list.append(temp_img)
        corresponding_tag_list.append(tag_list[index])

    return transformed_img_list, corresponding_tag_list


def balance_augment_data(img_list: list,
                         tag_list: list,
                         balance_data: bool,
                         augmentation_proportion: int,
                         verbose: int = 0,
                         ):
    if augmentation_proportion < 1:
        warnings.warn(f'augmentation_proportion = {augmentation_proportion} < 1,'
                      ' augmentation_proportion should be an integer >= 1.')
        print('Setting augmentation_proportion = 1.')
        augmentation_proportion = 1
    if verbose >= 1:
        print(f'Applying data balancing ({balance_data}) and augmentation ({augmentation_proportion}).'
              f' Num original obs: {len(img_list)}')

    balanced_augmented_img_list = []
    balanced_augmented_tag_list = []
    # balance data (and augment if required)
    if balance_data:
        imgs_by_class = {}
        for tag in global_constants.TREE_INFORMATION:
            imgs_by_class[tag] = []
        for img_index in range(len(img_list)):
            imgs_by_class[tag_list[img_index]].append(copy.deepcopy(img_list[img_index]))

        max_class_size = max([len(imgs_by_class[key]) for key in imgs_by_class])
        max_class_size_augmented = max_class_size * augmentation_proportion
        for class_index in imgs_by_class:
            imgs_this_class = imgs_by_class[class_index]
            proxy_tag_list = [class_index for _ in range(len(imgs_this_class))]
            class_augmentation_proportion = (max_class_size_augmented // len(imgs_this_class))
            temp_img_list = []
            temp_tag_list = []
            for i in range(class_augmentation_proportion - 1):
                new_img_list, new_tag_list = random_transform_img_list(
                    img_list=imgs_this_class,
                    tag_list=proxy_tag_list,
                    # apply_probability=0.6,
                )
                temp_img_list.extend(new_img_list)
                temp_tag_list.extend(new_tag_list)
            for tag in temp_tag_list:
                assert tag == class_index
            if verbose >= 2:
                print(f'\tclass: {class_index} ({global_constants.TREE_INFORMATION[class_index]["japanese_reading"]})')
                print(f'\tclass_augmentation_proportion: {class_augmentation_proportion}')
                print(f'\tNum original obs: {len(imgs_this_class)}')
                print(f'\tNum obs after augmentation: {len(temp_img_list) + len(imgs_this_class)}')
            balanced_augmented_img_list.extend(temp_img_list)
            balanced_augmented_tag_list.extend(temp_tag_list)
            balanced_augmented_img_list.extend(imgs_this_class)
            balanced_augmented_tag_list.extend(proxy_tag_list)
            assert len(balanced_augmented_img_list) == len(balanced_augmented_tag_list)

    # augment data without balancing
    if augmentation_proportion > 1 and not balance_data:
        temp_img_list = []
        temp_tag_list = []
        for i in range(augmentation_proportion - 1):
            new_img_list, new_tag_list = random_transform_img_list(
                img_list=img_list,
                tag_list=tag_list,
                # apply_probability=0.6,
            )
            temp_img_list.extend(new_img_list)
            temp_tag_list.extend(new_tag_list)
        balanced_augmented_img_list.extend(temp_img_list)
        balanced_augmented_tag_list.extend(temp_tag_list)
        balanced_augmented_img_list.extend(img_list)
        balanced_augmented_tag_list.extend(tag_list)
    if verbose >= 1:
        print(f'Data augmentation applied. Num obs after augmentation: {len(balanced_augmented_img_list)}')

    return balanced_augmented_img_list, balanced_augmented_tag_list
