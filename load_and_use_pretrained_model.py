import torchvision.transforms.functional as tf

import global_constants
from models import pretrained_regnet
from data_preprocessing import data_loading


verbose = 2
num_classes = len(global_constants.TREE_INFORMATION)
img_list, tag_list = data_loading.load_data(data_path=global_constants.STEP_2_DATA_PATH, verbose=verbose)
img = img_list[0]
print(f'img.shape: {img.shape}')
img = tf.to_tensor(img)
print(f'img.shape: {img.shape}')

model, preprocess = pretrained_regnet.get_regnet(training=True, num_classes=num_classes)
print(f'model:\n{model}')

# Apply inference preprocessing transforms
img = preprocess(img)
print(f'img.shape: {img.shape}')
batch = img.unsqueeze(0)
print(f'batch.shape: {batch.shape}')

# Use the model and print the predicted category
prediction = model(batch)
print(f'prediction.shape: {prediction.shape}')
prediction = prediction.squeeze(0).softmax(0)
print(f'prediction.shape: {prediction.shape}')
class_id = prediction.argmax().item()
print(f'class_id: {class_id}')
score = prediction[class_id].item()
category_name = global_constants.TREE_INFORMATION[class_id]['japanese_reading']
print(f"{category_name}: {100 * score:.1f}%")
print(f'true label: {tag_list[0]}')
