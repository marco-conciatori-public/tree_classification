import cv2
import copy
import random
import warnings

import global_constants
from data_preprocessing import image_utils


def random_transform_img(img, apply_probability: float = 0.5, show_img: bool = False):
    transformed_img = copy.deepcopy(img)
    # not clear if this is necessary
    # transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
    if random.uniform(0, 1) < apply_probability:
        transformed_img = cv2.flip(transformed_img, 1)
    if random.uniform(0, 1) < apply_probability:
        transformed_img = cv2.flip(transformed_img, 0)
    if random.uniform(0, 1) < apply_probability:
        transformed_img = cv2.rotate(transformed_img, cv2.ROTATE_90_CLOCKWISE)
    if random.uniform(0, 1) < apply_probability:
        transformed_img = cv2.rotate(transformed_img, cv2.ROTATE_180)
    if random.uniform(0, 1) < apply_probability:
        transformed_img = cv2.rotate(transformed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if random.uniform(0, 1) < apply_probability:
        transformed_img = image_utils.brightness(transformed_img, low=0.4, high=1.8)
    # if random.uniform(0, 1) < apply_probability:
    #     transformed_img = image_utils.channel_shift(transformed_img, value=10)
    if show_img:
        # show original image and transformed image in the same window
        show_img = cv2.hconcat([img, transformed_img])
        cv2.imshow(f'original - transformed', show_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return transformed_img


def random_transform_img_list(img_list: list,
                              tag_list: list,
                              apply_probability: float = 0.5,
                              show_img: bool = False,
                              ) -> (list, list):
    transformed_img_list = []
    corresponding_tag_list = []
    for index in range(len(img_list)):
        img = img_list[index]
        transformed_img = random_transform_img(img=img, apply_probability=apply_probability, show_img=show_img)
        transformed_img_list.append(transformed_img)
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
