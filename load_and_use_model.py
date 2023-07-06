import torch
import torchvision.transforms.functional as tf

import utils
import config
import global_constants
from models import model_utils
from data_preprocessing import data_loading


cpu = torch.device('cpu')

# PARAMETERS
verbose = 2
model_id = 0
partial_name = ''
jump = 50

img_list, tag_list = data_loading.load_data(data_path=global_constants.STEP_2_DATA_PATH, verbose=verbose)
print(f'img_list length: {len(img_list)}')

tree_class_counter = [0, 0, 0, 0, 0]
for tag in tag_list:
    tree_class_counter[tag] += 1
print(f'tree_class_counter: {tree_class_counter}')

model_path, info_path = utils.get_path_by_id(
    partial_name=partial_name,
    model_id=model_id,
    folder_path=global_constants.MODEL_OUTPUT_DIR,
)

loaded_model, meta_data = model_utils.load_model(
    model_path=model_path,
    device=cpu,
    training_mode=False,
    meta_data_path=info_path,
    verbose=verbose,
)


# get loss function from string name
loss_function = getattr(torch.nn, config.LOSS_FUNCTION_NAME)()

test_loss = 0.0
loss_counter = 0
targets = []
predictions = []
with torch.set_grad_enabled(False):
    for img_index in range(len(img_list)):
        if img_index % jump != 0:
            continue
        print(f'img_index: {img_index}')
        img = img_list[img_index]
        tensor_img = tf.to_tensor(img)
        tensor_img = tensor_img.unsqueeze(0)
        prediction = loaded_model(tensor_img)
        prediction = torch.nn.Softmax(dim=1)(prediction)
        prediction = prediction.squeeze(0)
        prediction = prediction.numpy()

        print('-------------------')
        for tree_class in range(len(prediction)):
            if prediction[tree_class] > config.TOLERANCE:
                print(f'\t{global_constants.TREE_INFORMATION[tree_class]["japanese_reading"]}: '
                      f'{round(prediction[tree_class] * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))}')
        print(f'\ttrue class: {global_constants.TREE_INFORMATION[tag_list[img_index]]["japanese_reading"]}')
