import torch


def compute(stat_scores: torch.Tensor) -> float:
    assert len(stat_scores.shape) == 2, f'stat_scores.shape: {stat_scores.shape}, expected (num_classes, num_inputs)'
    total_score = 0
    for class_stat_scores in stat_scores:
        tp, fp, tn, fn, support = class_stat_scores
        if (tp > 0) or (tn > 0):
            total_score += (tp + tn) / (tp + fp + tn + fn)
    return total_score / len(stat_scores)
