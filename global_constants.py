DATA_PATH = 'data/'
ONE_LEVEL_UP = '../'
STEP_1_DATA_PATH = DATA_PATH + 'step_1/'
STEP_2_DATA_PATH = DATA_PATH + 'step_2/'
STEP_3_DATA_PATH = DATA_PATH + 'step_3/'
OUTPUT_DIR = 'output/'
MODEL_OUTPUT_DIR = OUTPUT_DIR + 'models/'
PARAMETER_SEARCH_OUTPUT_DIR = OUTPUT_DIR + 'parameter_search/'
PARAMETER_SEARCH_FILE_NAME = 'configuration_and_results'
MAX_DECIMAL_PLACES = 4
EXTERNAL_PARAMETER_SEPARATOR = '-'
INTERNAL_PARAMETER_SEPARATOR = '_'
INFO_FILE_NAME = 'meta_data'
PYTORCH_FILE_EXTENSION = '.pt'
DL_FILE_NAME = 'data_loader'
ORTHOMOSAIC_DATA_PATH = DATA_PATH + 'orthomosaic/'

DECIDUOUS = 'deciduous'
EVERGREEN = 'evergreen'
TREE_INFORMATION = {
    0: {
        'japanese_reading': 'buna',
        'common_name': 'beech',
        'scientific_name': 'fagus crenata',
        'japanese_name': 'ブナ',
        'type': DECIDUOUS,
    },
    1: {
        'japanese_reading': 'matsu',
        'common_name': 'pine',
        'scientific_name': 'pinus spp',
        'japanese_name': 'マツ',
        'type': EVERGREEN,
    },
    2: {
        'japanese_reading': 'minekaede',
        'common_name': 'butterfly maple',
        'scientific_name': 'acer tschonoskii',
        'japanese_name': 'ミネカエデ',
        'type': DECIDUOUS,
    },
    3: {
        'japanese_reading': 'mizunara',
        'common_name': 'oak',
        'scientific_name': 'quescus crispula',
        'japanese_name': 'ミズナラ',
        'type': DECIDUOUS,
    },
    4: {
        'japanese_reading': 'nanakamado',
        'abbreviated_japanese_reading': 'nana',
        'common_name': 'japanese rowan',
        'scientific_name': 'sorbus commixta',
        'japanese_name': 'ナナカマド',
        'type': DECIDUOUS,
    },
    5: {
        'japanese_reading': 'kyaraboku',
        'common_name': 'japanese yew',
        'scientific_name': 'taxus cuspidata',
        'japanese_name': 'キャラボク',
        'type': EVERGREEN,
    },
    6: {
        'japanese_reading': 'inutsuge',
        'common_name': 'japanese holly',
        'scientific_name': 'ilex crenata',
        'japanese_name': 'イヌツゲ',
        'type': EVERGREEN,
    },
}

