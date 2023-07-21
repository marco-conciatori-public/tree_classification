import torch


def get_patch(img: torch.Tensor, size: int, top_left_coord: tuple):
    # for channel_first img
    x = top_left_coord[0]
    y = top_left_coord[1]
    patch = img[
            :,
            x : x + size,
            y : y + size,
    ].detach().clone()
    return patch
