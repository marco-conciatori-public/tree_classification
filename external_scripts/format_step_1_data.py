import cv2
from pathlib import Path

import global_constants as gc

input_folder = 'original/'
output_folder = 'step_1_8500/'

data_path = gc.DATA_PATH + input_folder
pure_path = Path(gc.ONE_LEVEL_UP + data_path)
print(f'pure_path: {pure_path}')
assert pure_path.exists(), f'Path {data_path} does not exist'
tif_list = []
# species_counter = {}

for file_path in pure_path.iterdir():
    if file_path.is_file():
        print(f'file_path: {file_path}')
        file_name = file_path.name
        print(f'file_name: {file_name}')
        new_file_name = file_name.lower()
        print(f'new_file_name: {new_file_name}')
        if new_file_name.startswith('nana_'):
            new_file_name = new_file_name.replace('nana_', 'nanakamado_')
            # print(f'new_file_name: {new_file_name}')
        new_file_path = f'{file_path.parent}\\{new_file_name}'
        print(f'new_file_path: {new_file_path}')
        file_path.rename(new_file_path)
        file_name = new_file_name
        print(f'actual file_path: {file_path}')

        # species_name, site_name, _ = file_name.split('_')
        # # print(f'species_name: {species_name}')
        # # print(f'site_name: {site_name}')
        # if site_name not in species_counter:
        #     species_counter[site_name] = {}
        # if species_name not in species_counter[site_name]:
        #     species_counter[site_name][species_name] = 0
        # species_counter[site_name][species_name] += 1
        # img = cv2.imread(str(file_path))
        # print(f'img.shape: {img.shape}')
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# select only .TIF files
folder_list = list(pure_path.glob('**/*.TIF'))
tif_list.extend(folder_list)
print(f'len(tif_list): {len(tif_list)}')
print(f'tif_list[0]: {tif_list[0]}')
print(f'total num patches: {len(tif_list)}')

# save images to the output data folder
save_folder_str = gc.ONE_LEVEL_UP + gc.DATA_PATH + output_folder
Path(save_folder_str).mkdir(parents=True, exist_ok=True)
print(f'save_folder_str: {save_folder_str}')
for tif_path in tif_list:
    img_new_path = str(save_folder_str) + '/' + tif_path.name
    # print(f'img_new_path: {save_path}')
    cv2.imwrite(img_new_path, cv2.imread(str(tif_path)))
