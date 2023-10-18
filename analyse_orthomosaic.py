import torch
import numpy as np
import tifffile as tifi
from pathlib import Path
import torchvision.transforms.functional as tf

import utils
import global_constants
from models import model_utils
from image_processing import orthomosaic_utils


# compute species probability distribution for each pixel in the image_processing image

# PARAMETERS
verbose = 2
model_id = 0
partial_name = 'regnet_y'
img_name = 'Zao1_211005.tif'
# in pixels
# set patch_size to None to use the crop_size from the model. Only works for torchvision pretrained models
# TODO: chiedere a quanti pixel corrisponde una patch reale di 2m x 2m
patch_size = None
stride = 128
# confidence prediction probability above which the prediction is considered valid
confidence_threshold = 0.6
img_name_no_extension = img_name.split('.')[0]
print(f'Orthomosaic used: {img_name_no_extension}')

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
    # print(f'patch_size from preprocess: {patch_size}')
else:
    assert patch_size is not None, 'patch_size must be specified if no custom_transforms are used'
    preprocess = None
print(f'network patch_size: {patch_size}')

# load orthomosaic image
orthomosaic_path = global_constants.ORTHOMOSAIC_DATA_PATH + img_name
orthomosaic = tifi.imread(orthomosaic_path)
# TODO: remove this line, only for speeding up testing
orthomosaic = orthomosaic[10000 : 12000, 10000 : 12000, :]
print(f'orthomosaic.shape: {orthomosaic.shape}')
# print(f'orthomosaic type: {type(orthomosaic)}')
# print(f'orthomosaic[0, 0, 0] type: {type(orthomosaic[0, 0, 0])}')
total_width = orthomosaic.shape[1]
total_height = orthomosaic.shape[0]
x_max = total_width - patch_size
y_max = total_height - patch_size

unknown_class_id = meta_data['num_classes']
num_classes_plus_unknown = meta_data['num_classes'] + 1

# initialize species distribution
# the last channel is used to count the number of times a pixel has been predicted
# the second to last channel is used to count the number of times a pixel has been predicted as unknown
species_distribution = np.zeros((total_height, total_width, num_classes_plus_unknown + 1), dtype=np.int8)

# remove fourth channel
orthomosaic = orthomosaic[ : , : , 0:3]
print(f'remove fourth/alpha channel orthomosaic.shape: {orthomosaic.shape}')

# change color space
# TODO: check if this is needed, probably it is not needed because probably the model is
#  trained with images without this conversion
# orthomosaic = cv2.cvtColor(orthomosaic, cv2.COLOR_BGR2RGB)
# print(f'change color space orthomosaic.shape: {orthomosaic.shape}')

# save_path = f'{global_constants.OUTPUT_DIR}{global_constants.ORTHOMOSAIC_FOLDER_NAME}{img_name_no_extension}/'
# cv2.imwrite(
#     f'{save_path}sub_img.png',
#     orthomosaic,
# )
# exit()

# to tensor
orthomosaic = tf.to_tensor(orthomosaic)
print(f'to tensor orthomosaic.shape: {orthomosaic.shape}')

# # to device
# image_processing = image_processing.to(device)
# print(f'to device {device}, image_processing type: {type(image_processing)}')

softmax = torch.nn.Softmax(dim=0)
# to limit the amount of ram used, one patch at a time is extracted from the orthomosaic image and fed to the model
for x in range(0, x_max, stride):
    for y in range(0, y_max, stride):
        # print(f'x: {x}, y: {y}')
        # extract patch
        patch = orthomosaic_utils.get_patch(img=orthomosaic, size=patch_size, top_left_coord=(x, y))
        # print(f'original patch.shape: {patch.shape}')

        # apply preprocessing
        if preprocess is not None:
            patch = preprocess(patch)
            # print(f'transformed patch.shape: {patch.shape}')
        # print(f'preprocessed patch.shape: {patch.shape}')
        # to_show_patch = patch.permute(1, 2, 0).numpy()
        # cv2.imshow('patch', to_show_patch)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # to device
        patch = patch.to(device)

        single_patch_batch = patch.unsqueeze(0)
        # print(f'single_patch_batch.shape: {single_patch_batch.shape}')
        single_prediction_batch = loaded_model(single_patch_batch)
        # print(f'single_prediction_batch.shape: {single_prediction_batch.shape}')
        prediction = single_prediction_batch.squeeze(0)
        # print(f'prediction.shape: {prediction.shape}')
        # print(f'prediction: {prediction}')
        prediction = softmax(prediction)
        # print(f'prediction softmax: {prediction}')
        prediction = prediction.detach().cpu().numpy()
        # print(f'prediction.detach().cpu().numpy(): {prediction}')
        predicted_class = prediction.argmax()
        # print(f'predicted_class: {predicted_class}')
        predicted_probability = prediction[predicted_class]
        # print(f'predicted_probability: {predicted_probability}')
        if predicted_probability < confidence_threshold:
            predicted_class = unknown_class_id
            # print('predicted_class unknown')

        # print(f'species_distribution[{x} : {x + patch_size}, {y} : {y + patch_size}, : ]: '
        #       f'{species_distribution[x : x + patch_size, y : y + patch_size, : ].shape}')
        # print(f'{species_distribution[x: x + 5, y: y + 5, : ].shape}')
        # print(f'{species_distribution[x: x + 5, y: y + 5, : ]}')
        species_distribution[
            x : x + patch_size,
            y : y + patch_size,
            predicted_class,
        ] += 1
        species_distribution[
            x:x + patch_size,
            y:y + patch_size,
            -1,
        ] += 1
        # print('species_distribution[x: x + patch_size, y: y + patch_size, : ] shape: ')
        # print(f'{species_distribution[x: x + patch_size, y: y + patch_size, : ].shape}')
        # print(f'{species_distribution[x: x + 5, y: y + 5, : ]}')

print('-------------------- end loop --------------------')
print(f'species_distribution.shape: {species_distribution.shape}')
print(f'species_distribution.type: {type(species_distribution)}')

# normalize species_distribution using the information in the last channel
for x in range(species_distribution.shape[0]):
    for y in range(species_distribution.shape[1]):
        for c in range(num_classes_plus_unknown):
            if species_distribution[x, y, c] > 0:
                species_distribution[x, y, c] = species_distribution[x, y, c] / species_distribution[x, y, -1]

# remove last channel/layer
species_distribution = species_distribution[ : , : , : -1]

print(f'species_distribution.shape: {species_distribution.shape}')
# print(f'species_distribution.type: {type(species_distribution)}')
# print(f'unique values count: {np.unique(species_distribution, return_counts=True)}')

assert species_distribution.max() <= 1, f'species_distribution.max() > 1. (max = {species_distribution.max()})'
assert species_distribution.min() >= 0, f'species_distribution.min() < 0. (min = {species_distribution.min()})'

# create one image for each class, where the alpha channel is the probability of that pixel being that class
save_path = f'{global_constants.OUTPUT_DIR}{global_constants.ORTHOMOSAIC_FOLDER_NAME}{img_name_no_extension}/'
Path(save_path).mkdir(parents=True, exist_ok=True)
for c in range(num_classes_plus_unknown):
    temp_img = np.zeros((total_height, total_width, 4), dtype=np.uint8)
    temp_img[ : , : , 3] = species_distribution[ : , : , c] * 255
    if c == unknown_class_id:
        colors = (0, 0, 0)
    else:
        colors = global_constants.TREE_INFORMATION[c]['display_color_rgb']
    temp_img[ : , : , 0] = colors[0]
    temp_img[ : , : , 1] = colors[1]
    temp_img[ : , : , 2] = colors[2]

    # print(f'temp_img.dtype: {temp_img.dtype}')
    tifi.imwrite(file=f'{save_path}{img_name_no_extension}_{c}.tif', data=temp_img)
    test_img = tifi.imread(f'{save_path}{img_name_no_extension}_{c}.tif')
    print(f'test_img.shape: {test_img.shape}')
    print(f'test_img.dtype: {test_img.dtype}')
