import numpy as np


def compute_classification_metrics(y_true, y_pred, num_classes):
    """
    Computes confusion matrix, accuracy, class-wise & macro precision, recall, and F-measure.
    y_true, y_pred: 1D arrays of integer class labels (0 to num_classes - 1).
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracy = np.trace(cm) / np.sum(cm)

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f_measure = np.zeros(num_classes)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision[c] + recall[c] > 0:
            f_measure[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])
        else:
            f_measure[c] = 0.0

    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f_measure = np.mean(f_measure)

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "mean_precision": mean_precision,
        "recall": recall,
        "mean_recall": mean_recall,
        "f_measure": f_measure,
        "mean_f_measure": mean_f_measure,
    }


def compute_regression_metrics(y_true, y_pred):
    """
    Computes RMSE and %RMSE.
    y_true, y_pred: 1D or 2D arrays of continuous values.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    variance = np.var(y_true)
    norm_factor = np.std(y_true) if variance > 1e-8 else (np.max(y_true) - np.min(y_true))
    pct_rmse = (rmse / norm_factor) * 100 if norm_factor != 0 else 0.0

    return {"rmse": rmse, "pct_rmse": pct_rmse}
