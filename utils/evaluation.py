import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

def evaluate_classifier(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def full_evaluation(y_true, y_pred, y_score):
    return evaluate_classifier(y_true, y_pred, y_score), confusion_matrix(y_true, y_pred), classification_report(
        y_true, y_pred, target_names=["No Deposit", "Deposit"], output_dict=True, zero_division=0)
