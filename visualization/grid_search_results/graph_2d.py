from pathlib import Path
import matplotlib.pyplot as plt

import global_constants


def plot_parameters(hp_to_plot: list,
                    excluded_key_list: list,
                    hp_evaluation: dict,
                    loss_set: str = 'test_loss',
                    rotate_x_labels: bool = False,
                    save_img: bool = False,
                    ):
    results = hp_evaluation['results']
    hp_to_plot = hp_to_plot[0]

    loss_list = []
    for el in results:
        loss_list.append(el[loss_set])

    title = f'Parameter Evaluation ({loss_set}) - Fixed parameters:\n'
    for key in hp_evaluation:
        if key not in excluded_key_list:
            title += f"{str(key).replace('_', ' ').title()}:" \
                     f" {str(hp_evaluation[key]).replace('_', ' ').replace('[', '').replace(']', '')}\n"

    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel(hp_to_plot)
    x_ticks_labels = []
    for el in hp_evaluation[hp_to_plot]:
        x_ticks_labels.append(round(el, global_constants.MAX_DECIMAL_PLACES))
    plt.ylabel('Loss Value')
    plt.title(title)
    if not rotate_x_labels:
        plt.xticks(
            ticks=range(len(loss_list)),
            labels=x_ticks_labels,
        )
    else:
        plt.xticks(
            ticks=range(len(loss_list)),
            labels=x_ticks_labels,
            rotation=35,
            horizontalalignment='right',
        )
    plt.tight_layout()
    if save_img:
        Path(global_constants.IMG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{global_constants.IMG_OUTPUT_DIR}param_2d_plot.png')
        print(f'Image "param_2d_plot.png" saved in "{global_constants.IMG_OUTPUT_DIR}"')
    else:
        plt.show()
    plt.close()
