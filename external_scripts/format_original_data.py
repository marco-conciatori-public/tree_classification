import cv2
import json
from pathlib import Path

import global_constants as gc

input_folder = 'paper_original_data/testing_data/plot_1/'
output_folder = 'paper_biodiversity_data/plot_1/'

original_data_path = gc.DATA_PATH + input_folder
pure_path = Path(gc.ONE_LEVEL_UP + original_data_path)
print(f'pure_path: {pure_path}')
assert pure_path.exists(), f'Path {original_data_path} does not exist'
tif_list = []
species_counter = {}
for site_path in pure_path.iterdir():
    print(f'site_path: {site_path}')
    site_name = site_path.name.lower()
    if site_name not in species_counter:
        species_counter[site_name] = {}
    for species_path in site_path.iterdir():
        print(f'species_path: {species_path}')
        species_name = species_path.name.lower()
        if species_name not in species_counter[site_name]:
            species_counter[site_name][species_name] = 0
        for file_path in species_path.iterdir():
            if file_path.is_file():
                # print(f'file_path: {file_path}')
                new_file_name = file_path.name.lower()
                # new_file_name = new_file_name.replace('-', '_')
                # print(f'new_file_name: {new_file_name}')
                new_file_path = f'{file_path.parent}\\{new_file_name}'
                # print(f'new_file_path: {new_file_path}')
                file_path.rename(new_file_path)
                # print(f'file_path.suffix: {file_path.suffix}')

            # count only TIFF files
            if file_path.suffix.lower() == '.tif':
                species_counter[species_name] += 1

print(f'species_counter: {species_counter}')
# select only .TIF files
folder_list = list(pure_path.glob('**/*.TIF'))
tif_list.extend(folder_list)
print(f'len(tif_list): {len(tif_list)}')
print(f'tif_list[0]: {tif_list[0]}')

print(f'total num patches: {len(tif_list)}')
species_counter['total_num_patches'] = len(tif_list)

# save images to the input data folder
save_folder_str = gc.ONE_LEVEL_UP + gc.DATA_PATH + output_folder
Path(save_folder_str).mkdir(parents=True, exist_ok=True)
print(f'save_folder_str: {save_folder_str}')
for tif_path in tif_list:
    img_new_path = str(save_folder_str) + '/' + tif_path.name
    # print(f'img_new_path: {save_path}')
    cv2.imwrite(img_new_path, cv2.imread(str(tif_path)))

# save meta_data info to json file
parts = output_folder.split('/')
parts = parts[:-1]
info_folder_str = gc.ONE_LEVEL_UP + gc.DATA_PATH + 'step_1_info/'
print(f'info_folder_str: {info_folder_str}')
print(f'parts: {parts}')
info_file_name = parts[-1] + '.json'
if len(parts) > 1:
    for part in parts[:-1]:
        info_folder_str += part + '/'
print(f'info_folder_str: {info_folder_str}')

Path(info_folder_str).mkdir(parents=True, exist_ok=True)
info_str = info_folder_str + info_file_name
print(f'info_str: {info_str}')
with open(info_str, 'w') as info_file:
    json.dump(species_counter, info_file)
