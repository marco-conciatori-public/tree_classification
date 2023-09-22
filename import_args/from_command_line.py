import argparse
import copy


def update_config(default_data_dict: dict) -> dict:
    parser = argparse.ArgumentParser()
    for key in default_data_dict:
        value = default_data_dict[key]
        if isinstance(value, list):
            parser.add_argument(f'--{key}', dest=key, type=type(value), nargs='*')
        # elif isinstance(value, dict):
        #     continue
        else:
            parser.add_argument(f'--{key}', dest=key, type=type(value))

    updated_data_dict = copy.deepcopy(default_data_dict)
    args = parser.parse_args()
    args_dict = vars(args)
    for key in args_dict:
        if args_dict[key] is not None:
            updated_data_dict[key] = args_dict[key]

    return updated_data_dict
