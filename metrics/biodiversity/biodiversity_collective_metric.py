import torchmetrics

from metrics.biodiversity import gini_simpson_index, shannon_wiener_index, species_richness


class BiodiversityCollectiveMetric(torchmetrics.Metric):
    """
    This class is used to store the information needed to compute all the biodiversity metrics.
    """
    def __init__(self, biodiversity_metric_names: list, class_information: dict, **kwargs):
        super().__init__()
        self.tag_list = []
        self.prediction_list = []
        self.biodiversity_metric_names = biodiversity_metric_names
        self.class_information = class_information
        self.log_base = 2
        if 'log_base' in kwargs:
            self.log_base = kwargs['log_base']

    def update(self, predicted_values, true_values) -> None:
        # predictions are in the form of a tensor of shape (batch_size, num_classes)
        #  where each element is the probability of the corresponding class
        # print(f'predicted_values.shape: {predicted_values.shape}')
        predicted_values = predicted_values.argmax(dim=1)
        # print(f'predicted_values.shape: {predicted_values.shape}')
        # send to cpu
        if hasattr(predicted_values, 'cpu'):
            true_values = true_values.cpu()
            predicted_values = predicted_values.cpu()
        # print(f'true_values.shape: {true_values.shape}')
        # print(f'predicted_values.shape: {predicted_values.shape}')

        self.tag_list.extend(true_values.detach().tolist())
        self.prediction_list.extend(predicted_values.detach().tolist())

    def compute(self):
        biodiversity_results = {}
        for metric_name in self.biodiversity_metric_names:
            biodiversity_results[metric_name] = {}

        if 'gini_simpson_index' in self.biodiversity_metric_names:
            biodiversity_results['gini_simpson_index']['true_result'] = gini_simpson_index.get_bio_diversity_index(
                tag_list=self.tag_list,
                class_information=self.class_information,
            )
            biodiversity_results['gini_simpson_index']['predicted_result'] = gini_simpson_index.get_bio_diversity_index(
                tag_list=self.prediction_list,
                class_information=self.class_information,
            )

        if 'shannon_wiener_index' in self.biodiversity_metric_names:
            biodiversity_results['shannon_wiener_index']['true_result'] = shannon_wiener_index.get_bio_diversity_index(
                tag_list=self.tag_list,
                class_information=self.class_information,
                log_base=self.log_base,
            )
            biodiversity_results['shannon_wiener_index']['predicted_result'] = shannon_wiener_index.get_bio_diversity_index(
                tag_list=self.prediction_list,
                class_information=self.class_information,
                log_base=self.log_base,
            )

        if 'species_richness' in self.biodiversity_metric_names:
            biodiversity_results['species_richness']['true_result'] = species_richness.get_bio_diversity_index(
                tag_list=self.tag_list
            )
            biodiversity_results['species_richness']['predicted_result'] = species_richness.get_bio_diversity_index(
                tag_list=self.prediction_list
            )


        return biodiversity_results
