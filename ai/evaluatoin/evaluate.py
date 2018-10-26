"""evaluation.evaluate
Core Evaluation Modules
"""

from sklearn import metrics as metrics

def compute_accuracy(prediction, ground_truth):
    """
    Return the accuracy score evaluated between PREDICTION and GROUND_TRUTH
    """
    return metrics.accuracy_score(
                prediction,
                ground_truth,
                normalize=True
            )

def compute_n_correct_samples(prediction, ground_truth):
    """
    Return the number of correctly classified samples between PREDICTION and GROUND_TRUTH
    """
    return metrics.accuracy_score(
                prediction,
                ground_truth,
                normalize=False
            )

def compute_f1(prediction, ground_truth, average="macro"):
    """
    Return the f1 score evaluated between PREDICTION and GROUND_TRUTH

    average - (default: "macro") mode of averaging for multi-class prediction
    """
    return metrics.f1_score(
                prediction,
                ground_truth,
                average=average
            )

def compute_confusion_matrix(prediction, ground_truth):
    """
    Return the confusion matrix evaluated between PREDICTION and GROUND_TRUTH
    """
    return metrics.confusion_matrix(
                prediction,
                ground_truth
            )

def compute_precision(prediction, ground_truth, average="macro"):
    """
    Return the precision score evaluated between PREDICTION and GROUND_TRUTH

    average - (default: "macro") mode of averaging for multi-class prediction
    """
    return metrics.precision_score(
                prediction,
                ground_truth,
                average=average
            )

def compute_recall(prediction, ground_truth, average="macro"):
    """
    Return the recall score evaluated between PREDICTION and GROUND_TRUTH

    average - (default: "macro") mode of averaging for multi-class prediction
    """
    return metrics.recall_score(
                prediction,
                ground_truth,
                average=average
            )

