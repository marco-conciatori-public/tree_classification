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

# observation = np.random.random_sample(size=(7, 6))
# observation = torch.tensor(observation).to(device)
# print('Observation/s:')
# print(observation)
# print(f'dimensions: {observation.shape}')
train_dl, val_dl, test_dl, img_shape = get_ready_data.get_data(
    shuffle=False,
    batch_size=1,
    train_val_test_proportions=config.TRAIN_VAL_TEST_PROPORTIONS,
    tolerance=config.TOLERANCE,
    augment_data=0,
    verbose=verbose,
)

print(f'len(test_dl): {len(test_dl)}')

# get loss function from string name
loss_function = getattr(torch.nn, config.LOSS_FUNCTION_NAME)()

test_loss = 0.0
loss_counter = 0
targets = []
predictions = []
with torch.set_grad_enabled(False):
    for obs, target in test_dl:
        prediction = loaded_model(obs)

        loss = loss_function(prediction, target)
        loss_counter += 1
        test_loss += loss

        targets.append(target)
        predictions.append(prediction)

print(f'len(targets): {len(targets)}')
print(f'len(predictions): {len(predictions)}')
test_loss = test_loss / loss_counter
print(f'loss_counter: {loss_counter}')
print(f'test_loss: {test_loss}')

print('----------------------------------')
test_loss, metric_evaluations = evaluation.eval(
    model=loaded_model,
    test_data=test_dl,
    loss_function_name=config.LOSS_FUNCTION_NAME,
    device=cpu,
    metrics=config.METRICS,
    verbose=verbose,
)

