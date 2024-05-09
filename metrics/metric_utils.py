import torch

import global_constants


def print_formatted_results(loss: float,
                            metrics: dict,
                            title: str = 'RESULTS',
                            ) -> None:
    print(title)
    print(f'- loss: {loss}')
    for metric_type in metrics:
        if len(metrics[metric_type]) == 0:
            continue
        print(f'- {metric_type}:')
        if metric_type == 'biodiversity':
            for metric_name in metrics[metric_type]:
                print(f'\t- {metric_name}:')
                result_dict = metrics[metric_type][metric_name]
                true_result = result_dict['true_result']
                predicted_result = result_dict['predicted_result']
                print(f'\t\t- ground truth: {format_value(value=true_result, as_percentage=False)}')
                print(f'\t\t- prediction:   {format_value(value=predicted_result, as_percentage=False)}')
        elif metric_type == 'classification':
            for metric_name in metrics[metric_type]:
                result = metrics[metric_type][metric_name]
                formatted_result = format_value(value=result, as_percentage=True)
                print(f'\t- {metric_name}: {formatted_result}')
        else:
            raise ValueError(f'unexpected metric type "{metric_type}"')


def format_value(value,
                 as_percentage: bool = False,
                 ) -> str:
    if torch.is_tensor(value):
        value = value.item()
    max_decimal_places = global_constants.MAX_DECIMAL_PLACES
    if as_percentage:
        max_decimal_places = max(global_constants.MAX_DECIMAL_PLACES - 2, 0)
        value = value * 100
    value = round(value, max_decimal_places)
    if as_percentage:
        return f'{value}%'
    return f'{value}'
