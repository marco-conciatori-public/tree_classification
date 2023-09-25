import torch
import cv2

import utils
import global_constants
from import_args import args
from models import model_utils
from data_preprocessing import data_loading


# import parameters
parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH)
model_id = 0
# partial_name = 'regnet_y_1_6'
partial_name = 'swin'
# use only those images, if None, use all images in folder
# img_name_list = ['buna_s1_0.tif']
img_name_list = None
jump = 1
use_targets = False
# data_path = global_constants.TO_PREDICT_FOLDER_PATH
data_path = global_constants.DATA_PATH + 'mixed_species/'

model_path, info_path = utils.get_path_by_id(
    partial_name=partial_name,
    model_id=model_id,
    folder_path=global_constants.MODEL_OUTPUT_DIR,
)
loaded_model, custom_transforms, meta_data = model_utils.load_model(
    model_path=model_path,
    device=parameters['device'],
    training_mode=False,
    meta_data_path=info_path,
    verbose=parameters['verbose'],
)

img_list, tag_list = data_loading.load_data(
    data_path=data_path,
    selected_names=img_name_list,
    use_targets=use_targets,
    verbose=parameters['verbose'],
)
print(f'img_list length: {len(img_list)}')
img_list = img_list[::jump]
print(f'img_list length: {len(img_list)}')
if use_targets:
    tag_list = tag_list[::jump]

# get loss function from string name
loss_function = getattr(torch.nn, parameters['loss_function_name'])()
softmax = torch.nn.Softmax(dim=0)
with torch.set_grad_enabled(False):
    for img_index in range(len(img_list)):
        img = img_list[img_index]

        # apply custom transforms
        if custom_transforms is not None:
            for transform in custom_transforms:
                img = transform(img)

        if img.device != parameters['device']:
            img = img.to(parameters['device'])

        # add batch dimension
        img = img.unsqueeze(0)
        prediction = loaded_model(img)
        prediction = prediction.squeeze(0)
        prediction = softmax(prediction)
        prediction = prediction.detach().cpu().numpy()
        top_class = prediction.argmax()

        print('-------------------')
        if use_targets:
            true_name = utils.get_tree_name(tag_list[img_index])
            print(f'TRUE LABEL: {true_name.upper()}')
        print('NETWORK EVALUATION:')
        for tree_class in range(len(prediction)):
            # if prediction[tree_class] >= config.TOLERANCE:
            text = f' - {utils.get_tree_name(tree_class)}: ' \
                   f'{round(prediction[tree_class] * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))} %'
            if tree_class == top_class:
                text = utils.to_bold_string(text)
            print(text)

        img = img_list[img_index]
        window_name = f'patch {img_index}'
        if use_targets:
            window_name = true_name
        # show image
        cv2.imshow(winname=window_name, mat=img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
