from pathlib import Path


def load_img(data_path, verbose: int = 0) -> list:
    pure_path = Path(data_path)
    assert pure_path.exists(), f'Path {data_path} does not exist.'
    assert pure_path.is_dir(), f'Path {data_path} is not a directory.'

    img_list = []
    for img_path in pure_path.iterdir():
        if img_path.is_file():
            img_list.append(img_path)

    if verbose >= 2:
        print(f'Loaded {len(img_list)} images from {data_path}.')

    return img_list
