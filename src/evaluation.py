import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score)

class ModelEvaluator:
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initializes the Evaluator with true labels, predicted labels, and optional probability scores.
        
        Parameters:
            y_true (array-like): True binary labels.
            y_pred (array-like): Predicted binary labels.
            y_pred_proba (array-like, optional): Predicted probability scores for the positive class.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

    def compute_classification_metrics(self):
        """
        Computes and prints common classification metrics.
        """
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("\nClassification Report:\n", classification_report(self.y_true, self.y_pred, zero_division=0))
        print("\nConfusion Matrix:\n", confusion_matrix(self.y_true, self.y_pred))

    def plot_roc_curve(self):
        """
        Plots the ROC curve and prints the AUC.
        """
        if self.y_pred_proba is None:
            print("No probability scores provided; cannot plot ROC curve.")
            return
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
        plt.plot([0, 1], [0, 1], 'r--', label='Random Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        print("ROC AUC:", roc_auc)

    def plot_precision_recall_curve(self):
        """
        Plots the Precision-Recall curve and prints the average precision.
        """
        if self.y_pred_proba is None:
            print("No probability scores provided; cannot plot Precision-Recall curve.")
            return
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_true, self.y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})', color='green')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall (PR) Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.show()
        print("Average Precision:", avg_precision)
