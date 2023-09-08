import torch
from torchvision.models import convnext


def replace_decision_layer(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    # Replace decision layer/s with a new, untrained layer that has num_classes outputs
    num_input_features = model.classifier[2].in_features

    # print(f'LayerNorm2d:\n{model.classifier[0]}')
    # print(f'Linear:\n{model.classifier[2]}')
    model.classifier[0] = convnext.LayerNorm2d(
        normalized_shape=(num_input_features,),
        eps=1e-06,
        elementwise_affine=True,
    )
    model.classifier[2] = torch.nn.Linear(in_features=num_input_features, out_features=num_classes, bias=True)
    model.classifier[2].requires_grad = True
    # print(f'LayerNorm2d:\n{model.classifier[0]}')
    # print(f'Linear:\n{model.classifier[2]}')

    return model
