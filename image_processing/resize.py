import cv2
import warnings


def get_min_dimension(img_list: list) -> int:
    min_dim = 1000000

    for img in img_list:
        try:
            width, height = img.shape[:2]
            if width < min_dim:
                min_dim = width
            if height < min_dim:
                min_dim = height
        except Exception as e:
            warnings.warn(f'WARNING: could not get shape of image. Exception: {e}')
            continue
    return min_dim


def resize_img_list(img_list: list, standard_img_dim: int = None, verbose: int = 0) -> list:
    resized_img_list = []
    if standard_img_dim is None:
        # get min width and height
        standard_img_dim = get_min_dimension(img_list)
    if verbose >= 2:
        print(f'resizing images to ({standard_img_dim}, {standard_img_dim})')

    for img in img_list:
        # resize images to standard_img_dim
        try:
            img = cv2.resize(img, (standard_img_dim, standard_img_dim))
        except Exception as e:
            warnings.warn(f'Error while resizing image: {e}')
            continue
        resized_img_list.append(img)
    return resized_img_list
