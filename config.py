VERBOSE: int = 2

# CONV_2D NN HYPERPARAMETERS
NUM_CONV_LAYERS: int = 3
DENSE_LAYERS: list = [4096, 512, 64]
CONVOLUTION_PARAMETERS: dict = {
    'in_channels': 3,
    'out_channels': 16,
    'groups': 1,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
}
POOLING_OPERATION: str = 'MaxPool2d'
POOLING_PARAMETERS: dict = {
    'kernel_size': 2,
    'stride': 2,
    'padding': 0,
}
MODEL_PARAMETERS: dict = {
    'num_conv_layers': NUM_CONV_LAYERS,
    'dense_layers': DENSE_LAYERS,
    'convolution_parameters': CONVOLUTION_PARAMETERS,
    'pooling_operation': POOLING_OPERATION,
    'pooling_parameters': POOLING_PARAMETERS,
}

# TRAINING HYPERPARAMETERS
BATCH_SIZE: int = 32
EPOCHS: int = 8
LEARNING_RATE: float = 0.01
OPTIMIZER_NAME: str = 'Adam'
LOSS_FUNCTION_NAME: str = 'CrossEntropyLoss'
SHUFFLE: bool = False
STANDARD_METRIC_ARGS: dict = {
    'task': 'multiclass',
    'num_classes': 5,
    'average': 'weighted',
    # 'average': 'macro',
    # 'average': 'micro',
}
METRICS: dict = {
    'Accuracy': STANDARD_METRIC_ARGS,
    'Precision': STANDARD_METRIC_ARGS,
    'Recall': STANDARD_METRIC_ARGS,
    'F1Score': STANDARD_METRIC_ARGS,
}


# must add up to 1 +/- TOLERANCE
TRAIN_VAL_TEST_PROPORTIONS: list = [0.8, 0.1, 0.1]
TOLERANCE: float = 0.01
# number of times the original dataset is duplicated with random variations
DATA_AUGMENTATION_PROPORTION: int = 20
