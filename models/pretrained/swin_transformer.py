import torch


def replace_decision_layer(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    # Replace decision layer/s with a new, untrained layer that has num_classes outputs
    num_input_features = model.head.in_features
    eps = model.norm.eps
    model.norm = torch.nn.LayerNorm(normalized_shape=(num_input_features,), eps=eps, elementwise_affine=True)
    model.head = torch.nn.Linear(in_features=num_input_features, out_features=num_classes, bias=True)

    return model
