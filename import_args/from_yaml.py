import yaml


def read_config(yaml_path) -> dict:
    with open(yaml_path) as f:
        data_dict = yaml.safe_load(f)

    return data_dict
