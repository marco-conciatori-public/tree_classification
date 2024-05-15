import global_constants as gc


def from_name(name: str) -> int:
    name_lower = name.lower()
    for key, value in gc.CLASS_INFORMATION.items():
        if 'abbreviated_japanese_romaji' in value:
            abbreviated_japanese_romaji = value['abbreviated_japanese_romaji']
            if abbreviated_japanese_romaji is not None:
                if abbreviated_japanese_romaji in name_lower:
                    return key
        if value['japanese_romaji'] in name_lower:
            return key

    raise ValueError(f'No tree found with image name "{name}"')


def convert_class_id_from_different_sets(class_id: int,
                                         class_information_source: dict,
                                         class_information_destination: dict,
                                         ) -> int:
    class_name = class_information_source[class_id][gc.SPECIES_LANGUAGE]
    for key, value in class_information_destination.items():
        if value[gc.SPECIES_LANGUAGE] == class_name:
            return key

    return -1
