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


def load_data(img_path_list: list, verbose: int = 0) -> list:
    data_list = []
    for img_path in img_path_list:
        # get class of each image
        img_class = get_class.from_name(Path(img_path).name)

        try:
            img = cv2.imread(img_path)
            data_list.append((img_class, img))
        except Exception as e:
            print(f'ERROR: could not load image {img_path}. Exception: {e}')

    return data_list
