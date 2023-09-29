import copy
import warnings


def update_config(default_data_dict: dict, **kwargs) -> dict:
    updated_data_dict = copy.deepcopy(default_data_dict)
    for key in kwargs:
        try:
            updated_data_dict[key] = kwargs[key]
        except KeyError:
            warnings.warn(f'Parameter "{key}" not found in the config file. Only parameters in the config file are '
                          f'valid arguments. The parameter "{key}" will be ignored.')

    return updated_data_dict
