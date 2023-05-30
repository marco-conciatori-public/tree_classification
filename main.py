import config
import global_constants
from data_preprocessing import data_loading


verbose = config.VERBOSE
img_list = data_loading.load_img(global_constants.DATA_PATH, verbose=verbose)
