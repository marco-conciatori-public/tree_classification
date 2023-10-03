import global_constants


def get_bio_diversity_index(dataset):
    # Species richness
    # number of different species in the chosen dataset
    # https://en.wikipedia.org/wiki/Species_richness
    num_classes = len(global_constants.TREE_INFORMATION)
    return num_classes
