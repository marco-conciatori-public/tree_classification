import torch
import torchvision.transforms.functional as tf

import utils
import config
import global_constants
from models import pretrained_regnet, training, evaluation
from data_preprocessing import data_loading, data_augmentation, custom_dataset


verbose = 2
num_classes = len(global_constants.TREE_INFORMATION)
augment_data = config.DATA_AUGMENTATION_PROPORTION
device = utils.get_available_device(verbose=verbose)

model, preprocess = pretrained_regnet.get_regnet(training=True, num_classes=num_classes)
model.to(device=device)
# print(f'model:\n{model}')

img_list, tag_list = data_loading.load_data(data_path=global_constants.PREPROCESSED_DATA_PATH, verbose=verbose)
print(f'img_list[0].shape: {img_list[0].shape}')
temp_img_list = []
for img in img_list:
    img = tf.to_tensor(img)
    img = preprocess(img)
    temp_img_list.append(img)
img_list = temp_img_list
print(f'img_list[0].shape: {img_list[0].shape}')

# apply data augmentation
temp_img_list = []
temp_tag_list = []
if augment_data > 1:
    for i in range(augment_data - 1):
        new_img_list, new_tag_list = data_augmentation.random_transform_img_list(
            img_list=img_list,
            tag_list=tag_list,
            # apply_probability=0.6,
        )
        temp_img_list.extend(new_img_list)
        temp_tag_list.extend(new_tag_list)
    img_list.extend(temp_img_list)
    tag_list.extend(temp_tag_list)

utils.check_split_proportions(train_val_test_proportions=config.TRAIN_VAL_TEST_PROPORTIONS, tolerance=config.TOLERANCE)

# create dataset
ds = custom_dataset.Dataset_from_obs_targets(
    obs_list=img_list,
    target_list=tag_list,
    # name='complete_dataset',
)
# split dataset
total_length = len(ds)
split_lengths = [int(total_length * proportion) for proportion in config.TRAIN_VAL_TEST_PROPORTIONS]
split_lengths[2] = total_length - split_lengths[0] - split_lengths[1]
train_ds, val_ds, test_ds = ds.random_split(lengths=split_lengths)

# create data loaders
train_dl = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE,
)
val_dl = torch.utils.data.DataLoader(
    dataset=val_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE,
)
test_dl = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE,
)

# get image shape
batched_img_tag = next(iter(train_dl))
batched_img_shape = batched_img_tag[0].shape
print(f'batched_img_shape: {batched_img_shape}.')
print(f'batched Target shape: {batched_img_tag[1].shape}.')
# remove batch dimension
img_shape = batched_img_shape[1:]
print(f'img_shape: {img_shape}.')

training_history = training.train(
    model=model,
    training_data=train_dl,
    validation_data=val_dl,
    epochs=config.EPOCHS,
    learning_rate=config.LEARNING_RATE,
    loss_function_name=config.LOSS_FUNCTION_NAME,
    optimizer_name=config.OPTIMIZER_NAME,
    device=device,
    verbose=verbose,
    save_model=True,
    save_path=global_constants.MODEL_OUTPUT_DIR,
    metrics=config.METRICS,
)
print(f'training_history:\n{training_history}')

test_loss, metric_evaluations = evaluation.eval(
    model=model,
    test_data=test_dl,
    loss_function_name=config.LOSS_FUNCTION_NAME,
    device=device,
    metrics=config.METRICS,
    verbose=verbose,
)
print(f'test_loss: {test_loss}')
print(f'test_metric_evaluations: {metric_evaluations}')
