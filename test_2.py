import torch

from metrics.metric_manager import MetricManager


target = torch.zeros(100)
target[0] = 1
print(f'shape target: {target.shape}')
print(target)
preds = torch.zeros(100, 2)
preds[:, 0] = 1
print(f'shape preds: {preds.shape}')
print(preds)
class_information = {1: 'class_1', 2: 'class_2'}

metric = MetricManager(
    biodiversity_metric_names=[],
    classification_metric_names=['accuracy', 'precision', 'recall', 'f1score'],
    class_information=class_information,
)
metric.update(predicted_probabilities=preds, true_values=target)
biodiversity_results, classification_results = metric.compute()
print('biodiversity_results')
print(biodiversity_results)
print('classification_results')
print(classification_results)
