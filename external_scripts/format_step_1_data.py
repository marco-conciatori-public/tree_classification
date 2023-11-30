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
        file_name = file_path.name
        print(f'file_name: {file_name}')
        new_file_name = file_name.lower()
        print(f'new_file_name: {new_file_name}')
        if new_file_name.startswith('nana_'):
            new_file_name = new_file_name.replace('nana_', 'nanakamado_')
            print(f'new_file_name: {new_file_name}')
        new_file_path = f'{file_path.parent}\\{new_file_name}'
        print(f'new_file_path: {new_file_path}')
        file_path.rename(new_file_path)
        file_name = new_file_name
        print(f'actual file_path: {file_path}')

        species_name = file_name.split('_')[0]
        if species_name not in species_counter:
            species_counter[species_name] = 0
        species_counter[species_name] += 1

print(f'species_counter: {species_counter}')
