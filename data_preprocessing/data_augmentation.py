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


def random_transform_img_list(data_list: list, apply_probability: float = 0.5):
    transformed_data_list = []
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

    for data in data_list:
        img = data[1]
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
        transformed_data_list.append((data[0], temp_img))

    return transformed_data_list
