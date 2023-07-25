import torch
import cv2

import utils
import config
import global_constants
from models import model_utils
from data_preprocessing import data_loading


# PARAMETERS
verbose = config.VERBOSE
model_id = 0
partial_name = 'regnet_y_1_6'
# use only those images, if None, use all images in folder
img_name_list = ['buna_s1_0.tif']
# img_name_list = None

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

img_list, _ = data_loading.load_data(
            data_path=global_constants.TO_PREDICT_FOLDER_PATH,
            selected_names=img_name_list,
            verbose=verbose,
        )
# print(f'img_list length: {len(img_list)}')

# get loss function from string name
loss_function = getattr(torch.nn, config.LOSS_FUNCTION_NAME)()
softmax = torch.nn.Softmax(dim=0)
with torch.set_grad_enabled(False):
    for img_index in range(len(img_list)):
        img = img_list[img_index]

        # apply custom transforms
        if custom_transforms is not None:
            for transform in custom_transforms:
                img = transform(img)

        if img.device != device:
            img = img.to(device)

        # add batch dimension
        img = img.unsqueeze(0)
        prediction = loaded_model(img)
        prediction = prediction.squeeze(0)
        prediction = softmax(prediction)
        prediction = prediction.detach().cpu().numpy()
        top_class = prediction.argmax()

        print('-------------------')
        # print(f'TRUE LABEL: '
        #       f'{global_constants.TREE_INFORMATION[shortened_tag_list[img_index]]["japanese_reading"].upper()}')
        print('NETWORK EVALUATION:')
        for tree_class in range(len(prediction)):
            if prediction[tree_class] >= config.TOLERANCE:
                print(f' - {global_constants.TREE_INFORMATION[tree_class]["japanese_reading"]}: '
                      f'{round(prediction[tree_class] * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))} %')

        img = img_list[img_index]
        # show image
        cv2.imshow(
            winname=global_constants.TREE_INFORMATION[top_class]["japanese_reading"],
            mat=img,
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
