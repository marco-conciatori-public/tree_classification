from metrics.biodiversity import biodiversity_metric_utils


def get_bio_diversity_index(tag_list: list):
    # Species richness
    # number of different species in the chosen dataset
    # https://en.wikipedia.org/wiki/Species_richness

    return biodiversity_metric_utils.get_number_of_species(tag_list)
