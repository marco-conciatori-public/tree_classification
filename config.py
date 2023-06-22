VERBOSE = 2

# CONV_2D NN HYPERPARAMETERS
NUM_CONV_LAYERS = 3
DENSE_LAYERS = [512, 256, 128, 64]
CONVOLUTION_PARAMETERS = {
    'in_channels': 3,
    'out_channels': 16,
    'groups': 1,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
}
POOLING_OPERATION = 'MaxPool2d'
POOLING_PARAMETERS = {
    'kernel_size': 2,
    'stride': 2,
    'padding': 0,
}
MODEL_PARAMETERS = {
    'num_conv_layers': NUM_CONV_LAYERS,
    'dense_layers': DENSE_LAYERS,
    'convolution_parameters': CONVOLUTION_PARAMETERS,
    'pooling_operation': POOLING_OPERATION,
    'pooling_parameters': POOLING_PARAMETERS,
}

# TRAINING HYPERPARAMETERS
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 0.001
OPTIMIZER_NAME = 'Adam'
LOSS_FUNCTION_NAME = 'CrossEntropyLoss'
SHUFFLE = False
METRICS = ['accuracy', 'precision', 'recall', 'f1']
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
TOLERANCE = 0.01
