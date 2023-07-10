import cv2
from pathlib import Path

import global_constants


def get_min_dimensions(img_list: list) -> (int, int):
    min_width = 1000000
    min_height = 1000000
    for img_path in img_list:
        img_path = str(img_path)
        try:
            img = cv2.imread(img_path)
        except Exception as e:
            print(f'ERROR: could not load image {img_path}. Exception: {e}')
            continue
        try:
            width, height = img.shape[:2]
            if width < min_width:
                min_width = width
            if height < min_height:
                min_height = height
        except Exception as e:
            print(f'ERROR: could not get shape of image {img_path}. Exception: {e}')
            continue
    return min_width, min_height


def resize_img(img_path, min_width: int, min_height: int):
    save_folder_path = Path(global_constants.STEP_2_DATA_PATH)
    if not save_folder_path.exists():
        save_folder_path.mkdir(parents=False)

    img_name = img_path.name
    img_path = str(img_path)
    try:
        img = cv2.imread(img_path)
    except Exception as e:
        print(f'ERROR: could not load image {img_path}. Exception: {e}')
        return
    try:
        img = cv2.resize(img, (min_width, min_height))
    except Exception as e:
        print(f'ERROR: could not resize image {img_path}. Exception: {e}')
        return
    # the [:-1] is used to remove due "\", that causes problems
    img_path = f'{global_constants.STEP_2_DATA_PATH}{img_name}'
    # print(f'Saving image to "{img_path}"')
    try:
        cv2.imwrite(img_path, img)
    except Exception as e:
        print(f'ERROR: could not save image {img_path}. Exception: {e}')
        return

    return
