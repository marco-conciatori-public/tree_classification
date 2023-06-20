DATA_PATH = 'data/'
ONE_LEVEL_UP = '../'
PREPROCESSED_DATA_PATH = DATA_PATH + 'preprocessed_data/'
INTERMEDIATE_DATA_PATH = DATA_PATH + 'intermediate_data/'
OUTPUT_DIR = 'output/'
MODEL_OUTPUT_DIR = OUTPUT_DIR + 'models/'
MAX_DECIMAL_PLACES = 4
EXTERNAL_PARAMETER_SEPARATOR = '-'
PYTORCH_FILE_EXTENTION = '.pt'
INFO_FILE_NAME = 'meta_data'

TREE_CATEGORIES_JAPANESE = ('buna', 'matsu', 'minekaede', 'mizunara', 'nanakamado')
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
        'scientific_name': 'pinus spp.',
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
}

