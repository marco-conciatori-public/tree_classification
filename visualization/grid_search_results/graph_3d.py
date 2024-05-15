import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import global_constants as gc


def plot_parameters(hp_to_plot: list,
                    excluded_key_list: list,
                    hp_evaluation: dict,
                    loss_set: str = 'test_loss',
                    bar_width: float = 0.75,
                    save_img: bool = False,
                    ):
    results = hp_evaluation['results']
    length = len(results)
    hp_1_name = hp_to_plot[0]
    hp_2_name = hp_to_plot[1]
    x_bar_coordinate = []
    # hp_1_dict
    #   key: hp_1 name (as a string)
    #   value: hp_1_index
    hp_1_dict = {}
    hp_1_index = 0
    # x width of the 3D bars
    x_width = np.ones(length) * bar_width

    y_bar_coordinate = []
    # hp_2_dict
    #   key: hp_2 name (as a string)
    #   value: hp_2_index
    hp_2_dict = {}
    hp_2_index = 0
    # y width of the 3D bars
    y_width = np.ones(length) * bar_width

    z_bar_coordinate = np.zeros(length)  # all bars start at 0 height
    # height of the bars (validation loss values)
    height = []

    # mean data for the following 2d plots
    hp_1_2d = {}
    hp_2_2d = {}

    total_loss = 0
    for el in results:
        hp_1_value = tuple(el[hp_1_name])
        if hp_1_value not in hp_1_dict:
            hp_1_dict[hp_1_value] = hp_1_index
            hp_1_index += 1
        x_bar_coordinate.append(hp_1_dict[hp_1_value])

        hp_2_value = el[hp_2_name]
        if hp_2_value not in hp_2_dict:
            hp_2_dict[hp_2_value] = hp_2_index
            hp_2_index += 1
        y_bar_coordinate.append(hp_2_dict[hp_2_value])

        loss = el[loss_set]
        height.append(loss)
        if hp_1_value not in hp_1_2d:
            hp_1_2d[hp_1_value] = 0
        hp_1_2d[hp_1_value] += loss
        if hp_2_value not in hp_2_2d:
            hp_2_2d[hp_2_value] = 0
        hp_2_2d[hp_2_value] += loss
        total_loss += loss

    for key in hp_1_2d:
        hp_1_2d[key] = hp_1_2d[key] / len(hp_2_2d)
    for key in hp_2_2d:
        hp_2_2d[key] = hp_2_2d[key] / len(hp_1_2d)
    mean_loss = total_loss / len(results)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x=x_bar_coordinate, y=y_bar_coordinate, z=z_bar_coordinate, dx=x_width, dy=y_width, dz=height)

    ax.set_xlabel(hp_1_name.title().replace('_', ' '), labelpad=15)
    ax.set_ylabel(hp_2_name.title().replace('_', ' '), labelpad=10)
    ax.set_zlabel('Loss Value')
    title = f'Parameter Evaluation ({loss_set}) - Fixed parameters:\n'
    for key in hp_evaluation:
        if key not in excluded_key_list:
            title += f"{str(key).replace('_', ' ').title()}:" \
                     f" {str(hp_evaluation[key]).replace('_', ' ').replace('[', '').replace(']', '')}\n"
    ax.set_title(title)
    hp_1_list = list(hp_1_dict.values())

    x_labels = []
    for el in hp_1_dict.keys():
        if isinstance(el, float):
            x_labels.append(f'{el:.2e}')
        else:
            x_labels.append(el)
    # plt.xticks(hp_1_list[::2], x_labels[::2])
    plt.xticks(hp_1_list, x_labels)
    plt.yticks(list(hp_2_dict.values()), list(hp_2_dict.keys()))
    plt.show()
    plt.close()

    plt.plot(range(len(hp_1_2d)), hp_1_2d.values())
    plt.xlabel(hp_1_name.title())
    plt.ylabel('Loss Value')
    plt.title(title)
    plt.xticks(ticks=range(len(hp_1_2d)),
               labels=hp_1_2d.keys(),
               rotation=35,
               horizontalalignment='right')
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.plot(range(len(hp_2_2d)), hp_2_2d.values())
    plt.xlabel(hp_2_name.title())
    plt.ylabel('Loss Value')
    plt.title(title)
    plt.xticks(ticks=range(len(hp_2_2d)),
               labels=hp_2_2d.keys(),
               rotation=35,
               horizontalalignment='right')
    plt.tight_layout()
    if save_img:
        Path(gc.IMG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{gc.IMG_OUTPUT_DIR}param_3d_plot.png')
        print(f'Image "param_3d_plot.png" saved in "{gc.IMG_OUTPUT_DIR}"')
    else:
        plt.show()
    plt.close()

    print('mean_loss:', mean_loss)
