import cv2
from pathlib import Path

import global_constants as gc

input_folder = 'original_unique_tree_patches/'
output_folder = 'step_1_unique_tree_patches/'

original_data_path = gc.DATA_PATH + input_folder
pure_path = Path(gc.ONE_LEVEL_UP + original_data_path)
print(f'pure_path: {pure_path}')
assert pure_path.exists(), f'Path {original_data_path} does not exist'
tif_list = []
# for site_path in pure_path.iterdir():
#     print(f'site_path: {site_path}')
#     for species_path in site_path.iterdir():
#         print(f'species_path: {species_path}')
#         if 'Nanakamado' in str(species_path):
#             counter = 0
#             for file_path in species_path.iterdir():
#                 if file_path.is_file():
#                     # print(f'file_path: {file_path}')
#                     new_file_name = f'nanakamado_{site_path.name}_{counter}'
#                     # print(f'new_file_name: {new_file_name}')
#                     new_file_path = f'{file_path.parent}\\{new_file_name}'
#                     # print(f'new_file_path: {new_file_path}')
#                     file_path.rename(new_file_path)
#                     counter += 1

# select only .TIF files
folder_list = list(pure_path.glob('**/*.TIF'))
tif_list.extend(folder_list)
print(f'len(tif_list): {len(tif_list)}')
print(f'tif_list[0]: {tif_list[0]}')

# save images to the input data folder
save_folder_path = Path(gc.ONE_LEVEL_UP + gc.DATA_PATH + output_folder)
if not save_folder_path.exists():
    save_folder_path.mkdir(parents=False)
# print(f'save_folder_path: {save_folder_path}')
for tif_path in tif_list:
    img_new_path = str(save_folder_path) + '/' + tif_path.name.lower()
    # print(f'img_new_path: {save_path}')
    cv2.imwrite(img_new_path, cv2.imread(str(tif_path)))
