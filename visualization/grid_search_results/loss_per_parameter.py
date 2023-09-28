from pathlib import Path
import matplotlib.pyplot as plt

import global_constants


def bar_plot(average_loss: dict, save_img: bool = False):
    counter = 0
    for parameter_name in average_loss:
        print(f'parameter_name: {parameter_name}')
        parameter_results = average_loss[parameter_name]
        for parameter_value in parameter_results:
            loss_value = parameter_results[parameter_value]
            if isinstance(parameter_value, tuple):
                parameter_value = parameter_value[1]
            print(f'parameter_value: {parameter_value}, loss_value: {loss_value}')
            plt.bar(
                x=counter,
                height=loss_value,
                # label=str(parameter_value),
            )
            counter += 1

    x_labels = []
    for parameter_name in average_loss:
        parameter_results = average_loss[parameter_name]
        for parameter_value in parameter_results:
            if isinstance(parameter_value, tuple):
                parameter_value = parameter_value[1]
            x_labels.append(str(parameter_name) + '=' + str(parameter_value))
    plt.xticks(
        ticks=range(counter),
        labels=x_labels,
        rotation=35,
        horizontalalignment='right',
    )
    plt.title('Average loss per parameter setting')
    # plt.ylim((0.3, 0.5))
    plt.tight_layout()
    if save_img:
        Path(global_constants.IMG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{global_constants.IMG_OUTPUT_DIR}loss_per_parameter.png')
        print(f'Image "loss_per_parameter.png" saved in "{global_constants.IMG_OUTPUT_DIR}"')
    else:
        plt.show()
    plt.close()
