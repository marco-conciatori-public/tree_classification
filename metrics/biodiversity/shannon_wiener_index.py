import math

from metrics.biodiversity import metric_utils


def get_bio_diversity_index(tag_list: list, class_information: dict, log_base=2):
    # Shannon-Wiener index
    # https://en.wikipedia.org/wiki/Diversity_index#Shannon_index

    proportion_of_trees_by_species = metric_utils.get_proportion_of_trees_by_species(
        tag_list=tag_list,
        class_information=class_information,
    )

    shannon_wiener_index = 0
    for tree in proportion_of_trees_by_species:
        p_i = proportion_of_trees_by_species[tree]
        shannon_wiener_index += p_i * math.log(p_i, log_base)

    shannon_wiener_index *= -1
    return shannon_wiener_index
