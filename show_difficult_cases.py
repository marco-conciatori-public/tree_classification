import torch
import cv2

import utils
import config
import global_constants
from models import model_utils
from data_preprocessing import get_ready_data, data_loading


cpu = torch.device('cpu')

# PARAMETERS
verbose = 2
model_id = 0
partial_name = 'regnety'
worst_n_predictions = 30

model_path, info_path = utils.get_path_by_id(
    partial_name=partial_name,
    model_id=model_id,
    folder_path=global_constants.MODEL_OUTPUT_DIR,
)

loaded_model, custom_transforms, meta_data = model_utils.load_model(
    model_path=model_path,
    device=cpu,
    training_mode=False,
    meta_data_path=info_path,
    verbose=verbose,
)

train_dl, val_dl, test_dl, img_shape = get_ready_data.get_data(
    shuffle=False,
    batch_size=1,
    balance_data=False,
    train_val_test_proportions=config.TRAIN_VAL_TEST_PROPORTIONS,
    # standard_img_dim=config.IMG_DIM,
    custom_transforms=custom_transforms,
    tolerance=config.TOLERANCE,
    augmentation_proportion=1,
    random_seed=config.RANDOM_SEED,
    verbose=verbose,
)

img_list = []
tag_list = []
for batch in train_dl:
    observation_batch, target_batch = batch
    img_list.append(observation_batch)
    tag_list.append(target_batch.squeeze(0).item())
for batch in val_dl:
    observation_batch, target_batch = batch
    img_list.append(observation_batch)
    tag_list.append(target_batch.squeeze(0).item())
for batch in test_dl:
    observation_batch, target_batch = batch
    img_list.append(observation_batch)
    tag_list.append(target_batch.squeeze(0).item())

# get loss function from string name
loss_function = getattr(torch.nn, config.LOSS_FUNCTION_NAME)()
softmax = torch.nn.Softmax(dim=0)
worst_predictions = []
with torch.set_grad_enabled(False):
    for img_index in range(len(img_list)):
        img = img_list[img_index]

        prediction = loaded_model(img)
        prediction = prediction.squeeze(0)
        prediction = softmax(prediction)
        prediction = prediction.numpy()
        top_class = prediction.argmax()

        true_class = tag_list[img_index]
        prediction_of_true_class = prediction[true_class]
        worst_predictions.append((prediction_of_true_class, img_index, prediction))

# sort worst_predictions
worst_predictions.sort(key=lambda x: x[0])
for i in range(worst_n_predictions):
    if i > worst_n_predictions:
        break
    prediction_of_true_class, img_index, prediction = worst_predictions[i]

    print('-------------------')
    print(f'TRUE LABEL: '
          f'{utils.get_tree_name(tag_list[img_index]).upper()}')
    print('NETWORK EVALUATION:')
    for tree_class in range(len(prediction)):
        if prediction[tree_class] >= config.TOLERANCE:
            print(f' - {utils.get_tree_name(tree_class)}: '
                  f'{round(prediction[tree_class] * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))}')

    # show image
    img = img_list[img_index]
    cv2.imshow(
        winname=utils.get_tree_name(tag_list[img_index]).upper(),
        mat=img,
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
