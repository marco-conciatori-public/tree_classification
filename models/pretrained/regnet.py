from torch import nn
# from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights
from torchvision import models

import utils
import global_constants


# also called Seer
def get_model(model_name: str, weights_name: str = 'DEFAULT', training: bool = False, num_classes: int = None):
    # biggest model: regnet_y_128gf
    #   best weights: IMAGENET1K_SWAG_E2E_V1
    # second biggest model: regnet_y_32gf
    # small model: regnet_y_1_6gf
    model_id = utils.get_available_id(partial_name=model_name, folder_path=global_constants.MODEL_OUTPUT_DIR)
    model_full_name = f'{model_name}{global_constants.INTERNAL_PARAMETER_SEPARATOR}Weights' \
                      f'{global_constants.EXTERNAL_PARAMETER_SEPARATOR}{weights_name}'

    # Initialize model with the best available weights
    # weights = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
    # model = regnet_y_128gf(weights=weights)
    weights = getattr(models, weights_name)
    model = getattr(models, model_name)(weights=weights)
    # Initialize the inference transforms
    preprocess = weights.transforms(antialias=True)
    model.name = model_full_name
    model.id = model_id

    if training:
        assert num_classes is not None, 'num_classes must be specified when training = True.'
        model.train()

        # Freeze all layers weights
        for param in model.parameters():
            param.requires_grad = False
        # Replace the last layer with a new, untrained layer that has num_classes outputs
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # unfreeze the last layer
        model.fc.requires_grad = True

    else:
        model.eval()
        # Freeze all layers weights
        for param in model.parameters():
            param.requires_grad = False

    # print(f'model:\n{model}')
    return model, preprocess
