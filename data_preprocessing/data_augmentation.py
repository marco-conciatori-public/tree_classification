import cv2
import copy
import random
import albumentations as ab
from albumentations.augmentations import transforms as ab_transforms


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


def apply_data_augmentation(img_list: list, tag_list: list, augment_data: int, verbose: int = 0):
    # apply data augmentation
    temp_img_list = []
    temp_tag_list = []
    if augment_data > 1:
        if verbose >= 1:
            print(f'Applying data augmentation. Num original obs: {len(img_list)}')
        for i in range(augment_data - 1):
            new_img_list, new_tag_list = random_transform_img_list(
                img_list=img_list,
                tag_list=tag_list,
                # apply_probability=0.6,
            )
            temp_img_list.extend(new_img_list)
            temp_tag_list.extend(new_tag_list)
        img_list.extend(temp_img_list)
        tag_list.extend(temp_tag_list)
        if verbose >= 1:
            print(f'Data augmentation applied. Num obs after augmentation: {len(img_list)}')

    return img_list, tag_list
