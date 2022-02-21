from analysis_tools.common import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score


def confusion_matrix_analysis(y_true, y_pred, figsize=figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    """Plot confusion matrix

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    figsize : tuple
        Figure size.

    dir_path : str
        Path to save the figure.

    plot : bool
        Whether to plot the figure or not.

    Returns
    -------
    collections of metrics : dictionary
        Confusion matrix, accuracy, precision, recall, f1 score

    """
    C = confusion_matrix(y_true, y_pred, normalize='all')
    fig, ax = plt.subplots(figsize=figsize)
    with FigProcessor(fig, dir_path, plot, "Confusion matrix"):
        sns.heatmap(C, annot=True, fmt='.3f', cmap='binary', ax=ax)
    return dict(
        confusion_matrix=C, accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred), recall=recall_score(y_true, y_pred),
        f1_score=f1_score(y_true, y_pred)
    )


def curve_analysis(y_true, y_score, figsize=figsize(2, 1), dir_path=PATH.RESULT, plot=PLOT):
    """Plot learning curve

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_score : array-like of shape (n_samples,)
        Estimated probabilities or output of a decision function.

    figsize : tuple
        Figure size.

    dir_path : str
        Path to save the figure.

    plot : bool
        Whether to plot the figure or not.
    """
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_score)
    fpr, tpr, thresholds_roc           = roc_curve(y_true, y_score)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    with FigProcessor(fig, dir_path, plot, "Learning Curves"):
        axes[0].plot(thresholds_pr, precisions[:-1], 'b--', label='Precision')
        axes[0].plot(thresholds_pr, recalls[:-1], 'g-', label='Recall')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Precision/Recall')
        axes[0].set_ylim([0, 1])
        axes[0].legend()

        axes[1].plot(recalls, precisions, label=f"PR-AUC: {average_precision_score(y_true, y_score):.3f}")
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].legend()

        axes[2].plot(fpr, tpr, linewidth=2, label=f"ROC-AUC: {roc_auc_score(y_true, y_score):.3f}")
        axes[2].plot([0, 1], [0, 1], 'k--')
        axes[2].set_xlabel('FPR(=FP/RealNegative)')
        axes[2].set_ylabel('TPR(Recall)')
        axes[2].set_xlim([0, 1])
        axes[2].set_ylim([0, 1])
        axes[2].legend()
