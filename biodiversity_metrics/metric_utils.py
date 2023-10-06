import global_constants


def get_total_num_trees(tag_list: list) -> int:
    return len(tag_list)


def get_num_trees_by_species(tag_list: list) -> dict:
    num_trees_by_species = {}
    for tree in global_constants.TREE_INFORMATION:
        if tree[global_constants.TREE_NAME_TO_SHOW] not in num_trees_by_species:
            num_trees_by_species[tree[global_constants.TREE_NAME_TO_SHOW]] = 0

    for tag in tag_list:
        num_trees_by_species[tag] += 1

    return num_trees_by_species


def get_proportion_of_trees_by_species(tag_list: list) -> dict:
    total_num_trees = get_total_num_trees(tag_list)
    num_trees_by_species = get_num_trees_by_species(tag_list)
    proportion_of_trees_by_species = {}
    for tree in num_trees_by_species:
        proportion_of_trees_by_species[tree] = num_trees_by_species[tree] / total_num_trees

    return proportion_of_trees_by_species
