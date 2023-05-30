from pathlib import Path

import global_constants

# DELETE FILSE BY EXTENTION
data_path = global_constants.DATA_PATH
pure_path = Path(global_constants.ONE_LEVEL_UP + data_path)
# print(f'pure_path: {pure_path}')
assert pure_path.exists(), f'Path {data_path} does not exist.'
for dir_path in pure_path.iterdir():
    if dir_path.is_dir():
        ovr_list = list(dir_path.glob('*.ovr'))
        for ovr_path in ovr_list:
            ovr_path.unlink()

        tfw_list = list(dir_path.glob('*.tfw'))
        for tfw_path in tfw_list:
            tfw_path.unlink()

        xml_list = list(dir_path.glob('*.xml'))
        for xml_path in xml_list:
            xml_path.unlink()


