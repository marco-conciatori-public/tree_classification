import torch


def replace_decision_layer(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    # Replace decision layer/s with a new, untrained layer that has num_classes outputs
    num_input_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_input_features, num_classes)
    # unfreeze the last layer
    model.fc.requires_grad = True

    return model
