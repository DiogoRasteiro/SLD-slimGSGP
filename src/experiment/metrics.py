"""Binary classification metrics: balanced accuracy, MCC, sensitivity, specificity, plus ROC AUC as a backup."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, recall_score, roc_auc_score


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    # AUC is a backup, not a main metric: if rare positives push every model to predict
    # mostly 0, the thresholded metrics can't separate models, but the ranking of raw scores can.
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    auc = float("nan")
    if y_score is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))
    return dict(
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        sensitivity=float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        specificity=float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        auc=auc,
        n_test=int(len(y_true)),
        n_test_positive=int(y_true.sum()),
        n_pred_positive=int(y_pred.sum()),
    )
