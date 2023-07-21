import matplotlib.pyplot as plt


def plot_parameters(hp_to_plot: list,
                    excluded_key_list: list,
                    hp_evaluation: dict,
                    loss_set: str = 'test_loss',
                    max_digits: int = 5,
                    rotate_x_labels: bool = False,
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
        x_ticks_labels.append(round(el, max_digits))
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
    plt.show()
    plt.close()
