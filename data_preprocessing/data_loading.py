import cv2
from pathlib import Path
from data_preprocessing import get_class


def load_img(data_path: str, verbose: int = 0) -> list:
    pure_path = Path(data_path)
    assert pure_path.exists(), f'Path "{data_path}" does not exist.'
    assert pure_path.is_dir(), f'Path "{data_path}" is not a directory.'

    img_list = []
    for img_path in pure_path.iterdir():
        if img_path.is_file():
            img_list.append(img_path)

    if verbose >= 2:
        print(f'Loaded {len(img_list)} images from "{data_path}".')

    return img_list


def load_data(img_folder_path: str, verbose: int = 0) -> (list, list):
    pure_path = Path(img_folder_path)
    assert pure_path.exists(), f'Path "{img_folder_path}" does not exist.'
    assert pure_path.is_dir(), f'Path "{img_folder_path}" is not a directory.'

    img_list = []
    tag_list = []
    for img_path in pure_path.iterdir():
        # get class of each image
        img_class = get_class.from_name(img_path.name)

        try:
            img = cv2.imread(str(img_path))
            img_list.append(img)
            tag_list.append(img_class)
        except Exception as e:
            print(f'ERROR: could not load image "{img_path}". Exception: {e}')

    return img_list, tag_list
