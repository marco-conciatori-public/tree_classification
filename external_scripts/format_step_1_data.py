import cv2
from pathlib import Path

import global_constants

folder = 'paper_nhung/'

data_path = global_constants.DATA_PATH + folder
pure_path = Path(global_constants.ONE_LEVEL_UP + data_path)
print(f'pure_path: {pure_path}')
assert pure_path.exists(), f'Path {data_path} does not exist'
tif_list = []
species_counter = {}

for file_path in pure_path.iterdir():
    if file_path.is_file():
        print(f'file_path: {file_path}')
        exit()
        file_name = file_path.name
        print(f'file_name: {file_name}')
        if file_name.startswith('nana_'):
            new_file_name = file_name.replace('nana_', 'nanakamado_')
            print(f'new_file_name: {new_file_name}')
            new_file_path = f'{file_path.parent}\\{new_file_name}'
            print(f'new_file_path: {new_file_path}')
            file_path.rename(new_file_path)
            file_name = new_file_name
            print(f'actual file_path: {file_path}')
            exit()

        species_name = file_name.split('_')[0]
        if species_name not in species_counter:
            species_counter[species_name] = 0
        species_counter[species_name] += 1

print(f'len(tif_list): {len(tif_list)}')
print(f'tif_list[0]: {tif_list[0]}')
print(f'species_counter: {species_counter}')

# # save images to the input data folder
# save_folder_path = Path(global_constants.ONE_LEVEL_UP + global_constants.DATA_PATH + output_folder)
# if not save_folder_path.exists():
#     save_folder_path.mkdir(parents=False)
# # print(f'save_folder_path: {save_folder_path}')
# for tif_path in tif_list:
#     img_new_path = str(save_folder_path) + '/' + tif_path.name.lower()
#     # print(f'img_new_path: {save_path}')
#     cv2.imwrite(img_new_path, cv2.imread(str(tif_path)))
