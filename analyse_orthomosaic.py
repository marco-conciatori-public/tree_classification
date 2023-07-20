import cv2
import warnings
import tifffile as tifi
import torchvision.transforms.functional as tf

import utils
import config
import global_constants
from models import model_utils
from data_preprocessing import image_utils


# compute species probability distribution for each pixel in the orthomosaic image

# PARAMETERS
verbose = 2
model_id = 0
partial_name = 'DEFAULT'
img_name = 'Zao1_211005.tif'
# in pixels
# set patch_size to None to use the crop_size from the model. Only works for torchvision pretrained models
patch_size = None
# TODO: chiedere a quanti pixel corrisponde una patch reale di 2m x 2m
stride = 128

# load model
device = utils.get_available_device(verbose=verbose)
model_path, info_path = utils.get_path_by_id(
    partial_name=partial_name,
    model_id=model_id,
    folder_path=global_constants.MODEL_OUTPUT_DIR,
)
loaded_model, custom_transforms, meta_data = model_utils.load_model(
    model_path=model_path,
    device=device,
    training_mode=False,
    meta_data_path=info_path,
    verbose=verbose,
)
if custom_transforms is not None:
    preprocess = custom_transforms[1]
    if patch_size is None:
        patch_size = preprocess.crop_size[0]
    print(f'patch_size from preprocess: {patch_size}')
else:
    assert patch_size is not None, 'patch_size must be specified if no custom_transforms are used'
    preprocess = None
print(f'preprocess: {preprocess}')

# load orthomosaic image
orthomosaic_path = global_constants.ORTHOMOSAIC_DATA_PATH + img_name
orthomosaic = tifi.imread(orthomosaic_path)
print(f'orthomosaic.shape: {orthomosaic.shape}')
total_width = orthomosaic.shape[1]
total_height = orthomosaic.shape[0]

# remove fourth channel
orthomosaic = orthomosaic[:, :, 0:3]
print(f'remove fourth channel orthomosaic.shape: {orthomosaic.shape}')

# change color space
orthomosaic = cv2.cvtColor(orthomosaic, cv2.COLOR_BGR2RGB)
print(f'change color space orthomosaic.shape: {orthomosaic.shape}')

# to tensor
orthomosaic = tf.to_tensor(orthomosaic)
print(f'to tensor orthomosaic.shape: {orthomosaic.shape}')

# to limit the amount of ram used, one patch at a time is extracted from the orthomosaic image and fed to the model
for x in range(0, total_width, stride):
    for y in range(0, total_height, stride):
        print(f'x: {x}, y: {y}')
        # extract patch
        patch = image_utils.get_patch(img=orthomosaic, size=patch_size, top_left_coord=(x, y))
        print(f'original patch.shape: {patch.shape}')
        # cv2.imshow('patch', patch)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # apply preprocessing
        if preprocess is not None:
            patch = preprocess(patch)
            print(f'transform patch.shape: {patch.shape}')

        exit()

