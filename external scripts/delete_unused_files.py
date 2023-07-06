from pathlib import Path
import cv2


import global_constants

# DELETE FILES BY EXTENSION
original_data_path = global_constants.DATA_PATH + 'original_data/'
pure_path = Path(global_constants.ONE_LEVEL_UP + original_data_path)
print(f'pure_path: {pure_path}')
assert pure_path.exists(), f'Path {original_data_path} does not exist.'
tif_list = []
for dir_path in pure_path.iterdir():
    # correct wrong 's1' to 's2' naming for Minekaede_s2 folder
    # also correct wrong '＿' to '_' in file names
    if dir_path.is_dir():
        if 'Minekaede_s2' in str(dir_path):
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    # print(f'file_path: {file_path}')
                    new_file_name = str(file_path.name).replace('_s1＿', '_s2_')
                    # print(f'new_file_name: {new_file_name}')
                    new_file_path = f'{file_path.parent}\\{new_file_name}'
                    # print(f'new_file_path: {new_file_path}')
                    file_path.rename(new_file_path)

    # select only .TIF files
    if dir_path.is_dir():
        folder_list = list(dir_path.glob('*.TIF'))
        tif_list.extend(folder_list)

# save images to the step_1_data folder
save_folder_path = Path(global_constants.ONE_LEVEL_UP + global_constants.STEP_1_DATA_PATH)
if not save_folder_path.exists():
    save_folder_path.mkdir(parents=False)
print(f'save_folder_path: {save_folder_path}')
for tif_path in tif_list:
    img_new_path = str(save_folder_path) + '/' + tif_path.name
    # print(f'img_new_path: {save_path}')
    cv2.imwrite(img_new_path, cv2.imread(str(tif_path)))
