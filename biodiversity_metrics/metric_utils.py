import numpy as np

import global_constants


def get_total_num_trees(tag_list: list) -> int:
    return len(tag_list)


def get_number_of_species(tag_list: list) -> int:
    tag_array = np.array(tag_list)
    num_classes = len(np.unique(tag_array))
    return num_classes


def get_unique_species(tag_list: list) -> list:
    tag_array = np.array(tag_list)
    class_indexes = np.unique(tag_array)
    class_names = []
    for class_index in class_indexes:
        class_names.append(global_constants.TREE_INFORMATION[class_index][global_constants.SPECIES_LANGUAGE])
    return class_names


def get_num_trees_by_species(tag_list: list) -> dict:
    num_trees_by_species = {}
    species_present = get_unique_species(tag_list)
    for tree_name in species_present:
        if tree_name not in num_trees_by_species:
            num_trees_by_species[tree_name] = 0

    for tag in tag_list:
        tree_class = global_constants.TREE_INFORMATION[tag]
        tree_name = tree_class[global_constants.SPECIES_LANGUAGE]
        num_trees_by_species[tree_name] += 1

    return num_trees_by_species


def get_proportion_of_trees_by_species(tag_list: list) -> dict:
    total_num_trees = get_total_num_trees(tag_list)
    num_trees_by_species = get_num_trees_by_species(tag_list)
    proportion_of_trees_by_species = {}
    for tree_name in num_trees_by_species:
        proportion_of_trees_by_species[tree_name] = num_trees_by_species[tree_name] / total_num_trees

    return proportion_of_trees_by_species
