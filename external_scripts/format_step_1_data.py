import cv2
from pathlib import Path

import global_constants

# folder = 'paper_nhung/'
folder = 'test/'

data_path = global_constants.DATA_PATH + folder
pure_path = Path(global_constants.ONE_LEVEL_UP + data_path)
print(f'pure_path: {pure_path}')
assert pure_path.exists(), f'Path {data_path} does not exist'
tif_list = []
species_counter = {}

for file_path in pure_path.iterdir():
    if file_path.is_file():
        print(f'file_path: {file_path}')
        file_name = file_path.name
        print(f'file_name: {file_name}')
        # new_file_name = file_name.lower()
        # print(f'new_file_name: {new_file_name}')
        # if new_file_name.startswith('nana_'):
        #     new_file_name = new_file_name.replace('nana_', 'nanakamado_')
        #     # print(f'new_file_name: {new_file_name}')
        # new_file_path = f'{file_path.parent}\\{new_file_name}'
        # print(f'new_file_path: {new_file_path}')
        # file_path.rename(new_file_path)
        # file_name = new_file_name
        # print(f'actual file_path: {file_path}')

        # species_name, site_name, _ = file_name.split('_')
        # # print(f'species_name: {species_name}')
        # # print(f'site_name: {site_name}')
        # if site_name not in species_counter:
        #     species_counter[site_name] = {}
        # if species_name not in species_counter[site_name]:
        #     species_counter[site_name][species_name] = 0
        # species_counter[site_name][species_name] += 1
        img = cv2.imread(str(file_path))
        print(f'img.shape: {img.shape}')
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print(f'species_counter: {species_counter}')
