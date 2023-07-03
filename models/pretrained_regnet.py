from torch import nn
from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights


# also Seer
def get_regnet(training: bool = False, num_classes: int = None):
    # Initialize model with the best available weights
    weights = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
    model = regnet_y_128gf(weights=weights)
    # Initialize the inference transforms
    preprocess = weights.transforms(antialias=True)

    if training:
        assert num_classes is not None, 'num_classes must be specified when training = True.'
        model.train()

        # Freeze all layers weights
        for param in model.parameters():
            param.requires_grad = False
        # Replace the last layer
        model.fc.out_features = num_classes
        # unfreeze the last layer
        model.fc.requires_grad = True

    else:
        model.eval()
        # Freeze all layers weights
        for param in model.parameters():
            param.requires_grad = False

    # print(f'model:\n{model}')
    return model, preprocess
