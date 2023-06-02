import global_constants


def from_name(name: str) -> int:
    name_lower = name.lower()
    for key, value in global_constants.TREE_INFORMATION.items():
        if value['japanese_reading'] in name_lower:
            return key

    raise ValueError(f'No tree found with image name "{name}".')