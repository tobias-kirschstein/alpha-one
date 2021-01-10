import numpy as np


def cross_entropy(targets, predictions, epsilon=1e-12):
    targets = np.array(targets)
    predictions = np.array(predictions)
    if len(targets.shape) == 1:
        targets = np.expand_dims(targets, 0)
    if len(predictions.shape) == 1:
        predictions = np.expand_dims(predictions, 0)

    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    cross_entropies = -np.sum(targets * np.log(predictions), axis=1)
    if len(cross_entropies == 1):
        return cross_entropies[0]
    else:
        return cross_entropies
