DATA_PATH = 'data/'
ONE_LEVEL_UP = '../'
STEP_1_DATA_PATH = DATA_PATH + 'step_1/'
STEP_2_DATA_PATH = DATA_PATH + 'step_2/'
STEP_3_DATA_PATH = DATA_PATH + 'step_3/'
OUTPUT_DIR = 'output/'
TO_PREDICT_FOLDER_NAME = 'to_predict/'
TO_PREDICT_FOLDER_PATH = DATA_PATH + TO_PREDICT_FOLDER_NAME
MODEL_OUTPUT_DIR = OUTPUT_DIR + 'models/'
PARAMETER_SEARCH_OUTPUT_DIR = OUTPUT_DIR + 'parameter_search/'
PARAMETER_SEARCH_FILE_NAME = 'configuration_and_results'
MAX_DECIMAL_PLACES = 4
EXTERNAL_PARAMETER_SEPARATOR = '-'
INTERNAL_PARAMETER_SEPARATOR = '_'
INFO_FILE_NAME = 'meta_data'
PYTORCH_FILE_EXTENSION = '.pt'
DL_FILE_NAME = 'data_loader'
ORTHOMOSAIC_FOLDER_NAME = 'orthomosaic/'
ORTHOMOSAIC_DATA_PATH = DATA_PATH + ORTHOMOSAIC_FOLDER_NAME

DECIDUOUS = 'deciduous'
EVERGREEN = 'evergreen'
TREE_INFORMATION = {
    0: {
        'japanese_reading': 'buna',
        'english_name': 'beech',
        'scientific_name': 'fagus crenata',
        'japanese_name': 'ブナ',
        'type': DECIDUOUS,
        'display_color_rgb': (0, 0, 255),
    },
    1: {
        'japanese_reading': 'matsu',
        'english_name': 'pine',
        'scientific_name': 'pinus spp',
        'japanese_name': 'マツ',
        'type': EVERGREEN,
        'display_color_rgb': (0, 255, 0),
    },
    2: {
        'japanese_reading': 'minekaede',
        'english_name': 'butterfly maple',
        'scientific_name': 'acer tschonoskii',
        'japanese_name': 'ミネカエデ',
        'type': DECIDUOUS,
        'display_color_rgb': (255, 0, 0),
    },
    3: {
        'japanese_reading': 'mizunara',
        'english_name': 'oak',
        'scientific_name': 'quescus crispula',
        'japanese_name': 'ミズナラ',
        'type': DECIDUOUS,
        'display_color_rgb': (255, 255, 0),
    },
    4: {
        'japanese_reading': 'nanakamado',
        'abbreviated_japanese_reading': 'nana',
        'english_name': 'japanese rowan',
        'scientific_name': 'sorbus commixta',
        'japanese_name': 'ナナカマド',
        'type': DECIDUOUS,
        'display_color_rgb': (0, 255, 255),
    },
    5: {
        'japanese_reading': 'kyaraboku',
        'english_name': 'japanese yew',
        'scientific_name': 'taxus cuspidata',
        'japanese_name': 'キャラボク',
        'type': EVERGREEN,
        'display_color_rgb': (255, 0, 255),
    },
    6: {
        'japanese_reading': 'inutsuge',
        'english_name': 'japanese holly',
        'scientific_name': 'ilex crenata',
        'japanese_name': 'イヌツゲ',
        'type': EVERGREEN,
        'display_color_rgb': (255, 255, 255),
    },
    7: {
        'japanese_reading': 'mizuki',
        'english_name': 'wedding cake tree',
        'scientific_name': 'cornus controversa',
        'japanese_name': 'ミズキ',
        'type': DECIDUOUS,
        'display_color_rgb': (255, 255, 255),  # TODO: change color
    },
    8: {
        'japanese_reading': 'koshiabura',
        'english_name': 'koshiabura',
        'scientific_name': 'chengiopanax sciadophylloides',
        'japanese_name': 'コシアブラ',
        'type': DECIDUOUS,
        'display_color_rgb': (255, 255, 255),  # TODO: change color
    },
}
TREE_NAME_TO_SHOW = 'japanese_reading'
