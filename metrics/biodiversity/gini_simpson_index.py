from metrics.biodiversity import metric_utils


def get_bio_diversity_index(tag_list: list, class_information: dict) -> float:
    # Gini-Simpson index
    # https://en.wikipedia.org/wiki/Diversity_index#Gini%E2%80%93Simpson_index

    proportion_of_trees_by_species = metric_utils.get_proportion_of_trees_by_species(
        tag_list=tag_list,
        class_information=class_information,
    )

    gini_simpson_index = 0
    for tree in proportion_of_trees_by_species:
        gini_simpson_index += proportion_of_trees_by_species[tree] ** 2

    gini_simpson_index = 1 - gini_simpson_index
    return gini_simpson_index
