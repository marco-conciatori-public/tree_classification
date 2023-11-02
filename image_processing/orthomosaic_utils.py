import torch
import numpy as np
import tifffile as tifi
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import global_constants


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


def save_class_layers(num_classes_plus_unknown: int,
                      unknown_class_id: int,
                      species_distribution: np.ndarray,
                      save_path: str,
                      ):
    # create one image for each class, where the alpha channel is the probability of that pixel being that class
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for class_index in range(num_classes_plus_unknown):
        temp_img = np.zeros((species_distribution.shape[0], species_distribution.shape[1], 4), dtype=np.uint8)
        temp_img[:, :, 3] = species_distribution[:, :, class_index] * 255
        colors = (0, 0, 0)
        class_name = 'unknown'
        if class_index != unknown_class_id:
            colors = global_constants.TREE_INFORMATION[class_index]['display_color_rgb']
            class_name = global_constants.TREE_INFORMATION[class_index][global_constants.SPECIES_LANGUAGE]
        temp_img[:, :, 0] = colors[0]
        temp_img[:, :, 1] = colors[1]
        temp_img[:, :, 2] = colors[2]

        tifi.imwrite(f'{save_path}{class_name}.tif', data=temp_img)


def save_effective_orthomosaic(orthomosaic: torch.Tensor, effective_max_x: int, effective_max_y: int, save_path: str):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    print(f'orthomosaic.shape: {orthomosaic.shape}')
    orthomosaic = orthomosaic.permute(1, 2, 0).numpy()
    print(f'orthomosaic shape colors last: {orthomosaic.shape}')
    orthomosaic = orthomosaic[: effective_max_x, : effective_max_y, :]
    print(f'orthomosaic effective shape: {orthomosaic.shape}')
    tifi.imwrite(
        f'{save_path}subset_img.tif',
        data=orthomosaic,
    )


def save_legend(save_path: str, num_classes_plus_unknown: int, unknown_class_id: int, expand=(0, 0, 0, 0)):
    # create a legend with the colors of the classes
    patches = []
    for c in range(num_classes_plus_unknown):
        color = [0, 0, 0]
        name = 'unknown'
        if c != unknown_class_id:
            name = global_constants.TREE_INFORMATION[c][global_constants.SPECIES_LANGUAGE]
            color = list(global_constants.TREE_INFORMATION[c]['display_color_rgb'])
            for rgb_index in range(3):
                color[rgb_index] = color[rgb_index] / 255
        patches.append(mpatches.Patch(color=color, label=name))
    legend = plt.legend(handles=patches)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f'{save_path}legend.png', dpi="figure", bbox_inches=bbox)
    plt.close()


def save_output(orthomosaic: torch.Tensor,
                species_distribution: np.ndarray,
                effective_max_x: int,
                effective_max_y: int,
                num_classes_plus_unknown: int,
                unknown_class_id: int,
                save_path: str,
                ):
    save_class_layers(
        num_classes_plus_unknown=num_classes_plus_unknown,
        unknown_class_id=unknown_class_id,
        species_distribution=species_distribution,
        save_path=save_path,
    )
    save_effective_orthomosaic(
        orthomosaic=orthomosaic,
        effective_max_x=effective_max_x,
        effective_max_y=effective_max_y,
        save_path=save_path,
    )
    save_legend(
        save_path=save_path,
        num_classes_plus_unknown=num_classes_plus_unknown,
        unknown_class_id=unknown_class_id,
    )
