import numpy as np


def get_bio_diversity_index(tag_list: list):
    # Species richness
    # number of different species in the chosen dataset
    # https://en.wikipedia.org/wiki/Species_richness

    tag_array = np.array(tag_list)
    num_classes = len(np.unique(tag_array))
    return num_classes
