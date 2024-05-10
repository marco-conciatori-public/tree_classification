import cv2
import warnings
from pathlib import Path

import global_constants
from data_preprocessing import get_class


def get_img_path_list(data_path: str, verbose: int = 0) -> list:
    pure_path = Path(data_path)
    assert pure_path.exists(), f'Path "{data_path}" does not exist'
    assert pure_path.is_dir(), f'Path "{data_path}" is not a directory'

    img_path_list = []
    for img_path in pure_path.iterdir():
        if img_path.is_file():
            img_path_list.append(img_path)

    if verbose >= 2:
        print(f'Loaded {len(img_path_list)} images from "{data_path}"')

    return img_path_list


def load_data(data_path: str,
              use_only_classes: list = None,
              model_class_information: dict = None,
              use_targets: bool = True,
              verbose: int = 0,
              ) -> (list, list, list):
    pure_path = Path(data_path)
    assert pure_path.exists(), f'Path "{data_path}" does not exist'
    assert pure_path.is_dir(), f'Path "{data_path}" is not a directory'
    if use_only_classes is not None and len(use_only_classes) > 0:
        assert use_targets, '"use_targets" must be True if "use_only_classes" is not None'

    img_list = []
    tag_list = []
    classes_found = []
    classes_use_only = []
    for img_path in pure_path.iterdir():
        try:
            img = cv2.imread(str(img_path))
            img_list.append(img)
        except Exception as e:
            print(f'ERROR: could not load image "{img_path}". Exception: {e}')

        if use_targets:
            # get class of each image
            class_id = get_class.from_name(img_path.name)
            tag_list.append(class_id)
            if class_id not in classes_found:
                classes_found.append(class_id)
            if use_only_classes is not None and len(use_only_classes) > 0:
                if class_id not in use_only_classes:
                    img_list.pop()
                    tag_list.pop()

                else:  # class_id in use_only_classes
                    if class_id not in classes_use_only:
                        classes_use_only.append(class_id)

    if use_only_classes is not None and len(use_only_classes) > 0:
        for class_id in use_only_classes:
            assert class_id in classes_found, f'class "{global_constants.CLASS_INFORMATION[class_id][global_constants.SPECIES_LANGUAGE]}" not found in chosen data "{data_path}"'
    else:
        classes_use_only = classes_found
    classes_use_only.sort()

    if verbose >= 2:
        print(f'Loaded {len(img_list)} images')
        print(f'classes_found: {classes_found.sort()}')
        print(f'intersection with use_only_classes: {classes_use_only}')

    if model_class_information is not None:
        for class_id in classes_use_only:
            class_name = global_constants.CLASS_INFORMATION[class_id][global_constants.SPECIES_LANGUAGE]
            class_in_model_classes = False
            for model_class_id in model_class_information:
                model_class_name = model_class_information[model_class_id][global_constants.SPECIES_LANGUAGE]
                if class_name == model_class_name:
                    class_in_model_classes = True
                    break
            if not class_in_model_classes:
                warnings.warn(f'the chosen model was not trained for class "{class_name}"')
        for model_class_id in model_class_information:
            model_class_name = model_class_information[model_class_id][global_constants.SPECIES_LANGUAGE]
            model_class_in_classes = False
            for class_id in classes_use_only:
                class_name = global_constants.CLASS_INFORMATION[class_id][global_constants.SPECIES_LANGUAGE]
                if class_name == model_class_name:
                    model_class_in_classes = True
                    break
            if not model_class_in_classes:
                for class_id in global_constants.CLASS_INFORMATION:
                    class_name = global_constants.CLASS_INFORMATION[class_id][global_constants.SPECIES_LANGUAGE]
                    if class_name == model_class_name:
                        classes_use_only.append(class_id)
                        break

    classes_use_only.sort()

    if verbose >= 2:
        print(f'added model known classes: {classes_use_only}')
        print('classes names: ', end='')
        for class_id in classes_use_only:
            print(f'{global_constants.CLASS_INFORMATION[class_id][global_constants.SPECIES_LANGUAGE]}', end=', ')
        print()

    # select relevant class information
    new_class_id = 0
    class_information = {}
    for class_id in classes_use_only:
        class_information[new_class_id] = global_constants.CLASS_INFORMATION[class_id]
        new_class_id += 1

    # update tag_list
    for i in range(len(tag_list)):
        tag_list[i] = classes_use_only.index(tag_list[i])

    return img_list, tag_list, class_information
