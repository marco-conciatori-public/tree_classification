import torch
import torchvision.transforms.functional as tf
import cv2
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import utils
import config
import global_constants
from models import model_utils
from data_preprocessing import data_loading


cpu = torch.device('cpu')

# PARAMETERS
verbose = 2
model_id = 3
partial_name = 'RegNet_Y_1_6'
jump = 50

img_list, tag_list = data_loading.load_data(data_path=global_constants.STEP_2_DATA_PATH, verbose=verbose)
print(f'img_list length: {len(img_list)}')

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

softmax = torch.nn.Softmax(dim=0)
top_predictions = []
with torch.set_grad_enabled(False):
    for img_index in range(len(img_list)):
        # if img_index % jump != 0:
        #     continue
        # print(f'img_index: {img_index}')
        img = img_list[img_index]
        # print(f'img shape: {img.shape}')
        # print(f'img type: {type(img)}')

        # show image
        # cv2.imshow(
        #     winname=global_constants.TREE_INFORMATION[tag_list[img_index]]["japanese_reading"].upper(),
        #     mat=img,
        # )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = tf.to_tensor(img)
        # print(f'img shape: {img.shape}')
        # print(f'img type: {type(img)}')
        img = img.unsqueeze(0)
        # print(f'img shape: {img.shape}')
        prediction = loaded_model(img)
        # print(f'prediction shape: {prediction.shape}')
        # print(f'prediction: {prediction}')
        prediction = prediction.squeeze(0)
        # print(f'prediction shape: {prediction.shape}')
        # print(f'prediction: {prediction}')
        prediction = softmax(prediction)
        # print(f'prediction shape: {prediction.shape}')
        # print(f'prediction: {prediction}')
        prediction = prediction.numpy()
        # print(f'prediction shape: {prediction.shape}')
        # print(f'prediction: {prediction}')
        top_class = prediction.argmax()
        top_predictions.append(top_class)
        # print(f'top_class: {top_class}')

        # print('-------------------')
        # print(f'\tTRUE LABEL: {global_constants.TREE_INFORMATION[tag_list[img_index]]["japanese_reading"].upper()}')
        # for tree_class in range(len(prediction)):
        #     if prediction[tree_class] >= config.TOLERANCE:
        #         print(f'\t{global_constants.TREE_INFORMATION[tree_class]["japanese_reading"]}: '
        #               f'{round(prediction[tree_class] * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))}')
        # exit()

# Plot the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_true=tag_list,
    y_pred=top_predictions,
    display_labels=global_constants.TREE_CATEGORIES_JAPANESE,
    xticks_rotation=60,
)
# sns.heatmap(cm,
#             annot=True,
#             fmt='g')
# plt.ylabel('Prediction',fontsize=13)
# plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix', fontsize=17)
plt.show()
