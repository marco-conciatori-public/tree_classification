import cv2
import tifffile as tifi

import utils
import config
import global_constants
from models import model_utils
from data_preprocessing import image_utils


# compute species probability distribution for each pixel in the orthomosaic image

# PARAMETERS
verbose = 2
model_id = 0
partial_name = 'regnet'
img_name = 'Zao1_211005.tif'

# load model
# device = utils.get_available_device(verbose=verbose)
# model_path, info_path = utils.get_path_by_id(
#     partial_name=partial_name,
#     model_id=model_id,
#     folder_path=global_constants.MODEL_OUTPUT_DIR,
# )
# loaded_model, custom_transforms, meta_data = model_utils.load_model(
#     model_path=model_path,
#     device=device,
#     training_mode=False,
#     meta_data_path=info_path,
#     verbose=verbose,
# )

# load orthomosaic image
img_path = global_constants.ORTHOMOSAIC_DATA_PATH + img_name
img = tifi.imread(img_path)
print(f'img.shape: {img.shape}')
patch = image_utils.get_patch(img=img, size=512, top_left_coord=(21000, 13000))
print(f'patch.shape: {patch.shape}')
cv2.imshow('patch', patch)
cv2.waitKey(0)
cv2.destroyAllWindows()

# extract patches
# patch_size = 256
# stride = 128
# patches = utils.extract_patches(img, patch_size, stride)

