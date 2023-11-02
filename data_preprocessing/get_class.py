import global_constants


def from_name(name: str) -> int:
    name_lower = name.lower()
    for key, value in global_constants.TREE_INFORMATION.items():
        if 'abbreviated_japanese_romaji' in value:
            abbreviated_japanese_romaji = value['abbreviated_japanese_romaji']
            if abbreviated_japanese_romaji is not None:
                if abbreviated_japanese_romaji in name_lower:
                    return key
        if value['japanese_romaji'] in name_lower:
            return key

    raise ValueError(f'No tree found with image name "{name}"')
