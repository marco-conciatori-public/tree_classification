import torch
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassStatScores
target = torch.zeros(100)
target[0] = 1
print(f'len(target): {len(target)}')
print(target)
preds = torch.zeros(100)
print(f'len(preds): {len(preds)}')
average_list = ['micro', 'macro', 'weighted', 'none']


def test_accuracy(stat_scores: torch.Tensor, _prefix: str = '\t') -> float:
    print(f'{_prefix}test_accuracy')
    if len(stat_scores.shape) > 1:
        total = 0
        for sub_stat_scores in stat_scores:
            print(f'\t{_prefix}sub_stat_scores: {sub_stat_scores}')
            partial_result = test_accuracy(sub_stat_scores, _prefix=_prefix+'\t')
            total += partial_result
            print(f'\t{_prefix}partial_result: {partial_result}')
        return total / len(stat_scores)
    else:
        tp, fp, tn, fn, support = stat_scores
        return (tp + tn) / (tp + fp + tn + fn)


def test_precision(stat_scores: torch.Tensor, _prefix: str = '\t') -> float:
    print(f'{_prefix}test_precision')
    if len(stat_scores.shape) > 1:
        total = 0
        for sub_stat_scores in stat_scores:
            print(f'\t{_prefix}sub_stat_scores: {sub_stat_scores}')
            partial_result = test_precision(sub_stat_scores, _prefix=_prefix+'\t')
            total += partial_result
            print(f'\t{_prefix}partial_result: {partial_result}')
        return total / len(stat_scores)
    else:
        tp, fp, tn, fn, support = stat_scores
        return tp / (tp + fp)


def test_recall(stat_scores: torch.Tensor, _prefix: str = '\t') -> float:
    print(f'{_prefix}test_recall')
    if len(stat_scores.shape) > 1:
        total = 0
        for sub_stat_scores in stat_scores:
            print(f'\t{_prefix}sub_stat_scores: {sub_stat_scores}')
            partial_result = test_recall(sub_stat_scores, _prefix=_prefix+'\t')
            total += partial_result
            print(f'\t{_prefix}partial_result: {partial_result}')
        return total / len(stat_scores)
    else:
        tp, fp, tn, fn, support = stat_scores
        return tp / (tp + fn)


def test_f1(stat_scores: torch.Tensor, _prefix: str = '\t') -> float:
    print(f'{_prefix}test_f1')
    if len(stat_scores.shape) > 1:
        total = 0
        for sub_stat_scores in stat_scores:
            print(f'\t{_prefix}sub_stat_scores: {sub_stat_scores}')
            partial_result = test_f1(sub_stat_scores, _prefix=_prefix+'\t')
            total += partial_result
            print(f'\t{_prefix}partial_result: {partial_result}')
        return total / len(stat_scores)
    else:
        tp, fp, tn, fn, support = stat_scores
        return 2 * tp / (2 * tp + fp + fn)


for average in average_list:
    print(f'average: {average}')
    accuracy_func = MulticlassAccuracy(num_classes=2, average=average)
    precision_func = MulticlassPrecision(num_classes=2, average=average)
    recall_func = MulticlassRecall(num_classes=2, average=average)
    f1_func = MulticlassF1Score(num_classes=2, average=average)
    stat_scores_func = MulticlassStatScores(num_classes=2, average=average)
    accuracy_result = accuracy_func(preds, target)
    precision_result = precision_func(preds, target)
    recall_result = recall_func(preds, target)
    f1_result = f1_func(preds, target)
    stat_scores_result = stat_scores_func(preds, target)
    test_accuracy_result = test_accuracy(stat_scores_result)
    test_precision_result = test_precision(stat_scores_result)
    test_recall_result = test_recall(stat_scores_result)
    test_f1_result = test_f1(stat_scores_result)
    print(f'\tstat_scores_result:   {stat_scores_result}')
    print('\t                            TP,     FP,     TN,     FN,     support')
    print(f'\taccuracy_result:      {accuracy_result}')
    print(f'\ttest_accuracy_result: {test_accuracy_result}')
    print(f'\tprecision_result:     {precision_result}')
    print(f'\ttest_precision_resul: {test_precision_result}')
    print(f'\trecall_result:        {recall_result}')
    print(f'\ttest_recall_result:   {test_recall_result}')
    print(f'\tf1_result:            {f1_result}')
    print(f'\ttest_f1_result:       {test_f1_result}')
    print('------------------------------------------------------')
    print()
