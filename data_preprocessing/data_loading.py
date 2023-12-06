import cv2
from pathlib import Path

from data_preprocessing import get_class
import global_constants


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
    classes_used = []
    for img_path in pure_path.iterdir():
        try:
            img = cv2.imread(str(img_path))
            img_list.append(img)
        except Exception as e:
            print(f'ERROR: could not load image "{img_path}". Exception: {e}')

        if use_targets:
            # get class of each image
            img_class = get_class.from_name(img_path.name)
            tag_list.append(img_class)
            if img_class not in classes_found:
                classes_found.append(img_class)
            if use_only_classes is not None and len(use_only_classes) > 0:
                if img_class not in use_only_classes:
                    img_list.pop()
                    tag_list.pop()

                else:  # img_class in use_only_classes
                    if img_class not in classes_used:
                        classes_used.append(img_class)

    if use_only_classes is not None and len(use_only_classes) > 0:
        for img_class in use_only_classes:
            assert img_class in classes_found, f'Class "{img_class}" not found in chosen data "{data_path}"'
    else:
        classes_used = classes_found
    classes_used.sort()

    if verbose >= 2:
        print(f'Loaded {len(img_list)} images')
        print(f'classes_found: {classes_found}')
        print(f'classes_used: {classes_used}')

    # select relevant class information
    new_class_id = 0
    class_information = {}
    for class_id in classes_used:
        class_information[new_class_id] = global_constants.CLASS_INFORMATION[class_id]
        new_class_id += 1

    # update tag_list
    for i in range(len(tag_list)):
        tag_list[i] = classes_used.index(tag_list[i])

    return img_list, tag_list, class_information
