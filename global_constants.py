DATA_PATH = 'data/'
ONE_LEVEL_UP = '../'
STEP_2_DATA_PATH = DATA_PATH + 'step_2/'
OUTPUT_DIR = 'output/'
TO_PREDICT_FOLDER_NAME = 'to_predict/'
TO_PREDICT_FOLDER_PATH = DATA_PATH + TO_PREDICT_FOLDER_NAME
MODEL_OUTPUT_DIR = OUTPUT_DIR + 'models/'
IMG_OUTPUT_DIR = OUTPUT_DIR + 'img/'
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
CONFIG_PARAMETER_PATH = 'config.yaml'

DECIDUOUS = 'deciduous'
EVERGREEN = 'evergreen'
TREE_INFORMATION = {
    0: {
        'japanese_romaji': 'buna',
        'english': 'beech',
        'latin': 'fagus crenata',
        'japanese': 'ブナ',
        'type': DECIDUOUS,
        'display_color_rgb': (0, 255, 0),
    },
    1: {
        'japanese_romaji': 'matsu',
        'english': 'pine',
        'latin': 'pinus spp',
        'japanese': 'マツ',
        'type': EVERGREEN,
        'display_color_rgb': (255, 0, 255),
    },
    2: {
        'japanese_romaji': 'minekaede',
        'english': 'butterfly maple',
        'latin': 'acer tschonoskii',
        'japanese': 'ミネ楓',
        'type': DECIDUOUS,
        'display_color_rgb': (0, 127.5, 0),
    },
    3: {
        'japanese_romaji': 'mizunara',
        'english': 'oak',
        'latin': 'quercus crispula',
        'japanese': 'ミズナラ',
        'type': DECIDUOUS,
        'display_color_rgb': (255, 127.5, 0),
    },
    4: {
        'japanese_romaji': 'nanakamado',
        'abbreviated_japanese_romaji': 'nana',
        'english': 'japanese rowan',
        'latin': 'sorbus commixta',
        'japanese': 'ナナカマド',
        'type': DECIDUOUS,
        'display_color_rgb': (127.5, 191.25, 127.5),
    },
    5: {
        'japanese_romaji': 'kyaraboku',
        'english': 'japanese yew',
        'latin': 'taxus cuspidata',
        'japanese': 'キャラボク',
        'type': EVERGREEN,
        'display_color_rgb': (115.79, 115.79, 147.34),
    },
    6: {
        'japanese_romaji': 'inutsuge',
        'english': 'japanese holly',
        'latin': 'ilex crenata',
        'japanese': 'イヌツゲ',
        'type': EVERGREEN,
        'display_color_rgb': (253.4, 126.06, 199.52),
    },
    7: {
        'japanese_romaji': 'mizuki',
        'english': 'wedding cake tree',
        'latin': 'cornus controversa',
        'japanese': 'ミズキ',
        'type': DECIDUOUS,
        'display_color_rgb': (46.99, 254.31, 251.28),
    },
    8: {
        'japanese_romaji': 'koshiabura',
        'english': 'koshiabura',
        'latin': 'chengiopanax sciadophylloides',
        'japanese': 'コシアブラ',
        'type': DECIDUOUS,
        'display_color_rgb': (255, 0, 0),
    },
    9: {
        # 'japanese_romaji': 'momi',
        'japanese_romaji': 'fir',
        'english': 'fir',
        'latin': 'abies ?',
        'japanese': 'モミ',
        'type': EVERGREEN,
        'display_color_rgb': (0, 0, 255),
    },
}
SPECIES_LANGUAGE = 'japanese_romaji'
