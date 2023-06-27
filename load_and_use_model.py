import torch

import config
import utils
import global_constants
from models import model_utils, evaluation
from data_preprocessing import get_ready_data


cpu = torch.device('cpu')

# PARAMETERS
verbose = 2
model_id = 0
partial_name = ''

img_list = []

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
    for img in img_list:
        tensor_img = torch.tensor(img).to(cpu)
        prediction = loaded_model(tensor_img)

        print('-------------------')
        for tree_class in range(len(prediction)):
            if prediction[tree_class] > config.TOLERANCE:
                print(f'\t{global_constants.TREE_INFORMATION[tree_class]["japanese_reading"]}: '
                      f'{round(prediction[tree_class] * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))}')
        print(f'\ttrue class: {global_constants.TREE_INFORMATION[tag_list[img_index]]["japanese_reading"]}')
