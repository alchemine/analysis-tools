"""Metric analysis tools

Performance evaluation metrics are defined here.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score


### Classification
def confusion_matrix_analysis(y_true, y_pred, dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
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

    show_plot : bool
        Whether to show the figure.

    Returns
    -------
    collections of metrics : dictionary
        Confusion matrix, accuracy, precision, recall, f1 score

    Examples
    --------
    >>> from analysis_tools.metrics import confusion_matrix_analysis
    >>> y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 0, 1, 1, 1, 1, 1]
    >>> confusion_matrix_analysis(y_true, y_pred, dir_path='.')
    """
    normalized_C = confusion_matrix(y_true, y_pred, normalize='true')
    assert all(normalized_C.sum(axis=1) == 1), "Confusion matrix is not normalized"

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    with FigProcessor(fig, dir_path, show_plot, "Confusion matrix"):
        sns.heatmap(normalized_C, annot=False, fmt='.2%', cmap='gray', ax=axes[0, 0])
        sns.heatmap(normalized_C, annot=True, fmt='.2%', cmap='gray', ax=axes[0, 1])

        normalized_C_off_diagonal = copy(normalized_C)
        np.fill_diagonal(normalized_C_off_diagonal, 0)  # off-diagonal
        sns.heatmap(normalized_C_off_diagonal, annot=False, fmt='.2%', cmap='gray', ax=axes[1, 0])
        sns.heatmap(normalized_C_off_diagonal, annot=True, fmt='.2%', cmap='gray', ax=axes[1, 1])
        for ax in axes.flat:
            ax.xaxis.tick_top()
    return dict(
        confusion_matrix=normalized_C,
        accuracy=accuracy_score(y_true, y_pred), precision=precision_score(y_true, y_pred), recall=recall_score(y_true, y_pred), f1_score=f1_score(y_true, y_pred),
    )
def curve_analysis(y_true, y_score,           dir_path=None, figsize=FIGSIZE, show_plot=SHOW_PLOT):
    """Plot Precision-Recall and ROC curves

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_score : array-like of shape (n_samples,)
        Estimated probabilities or output of a decision function.

    dir_path : str
        Path to save the figure.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the figure.

    Examples
    --------
    >>> from analysis_tools.metrics import curve_analysis
    >>> y_true  = [0, 0, 0, 0, 1, 1, 1, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8, 0.85, 0.8, 0.9, 0.95]
    >>> curve_analysis(y_true, y_score, dir_path='.')
    """
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_score)
    fpr, tpr, thresholds_roc           = roc_curve(y_true, y_score)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    with FigProcessor(fig, dir_path, show_plot, "Precision-Recall & ROC curves"):
        # Thresholds-PR
        axes[0].show_plot(thresholds_pr, precisions[:-1], 'b--', label='Precision')
        axes[0].show_plot(thresholds_pr, recalls[:-1], 'g-', label='Recall')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Precision/Recall')
        axes[0].set_ylim([0, 1])
        axes[0].legend()

        # Precision-Recall
        axes[1].show_plot(recalls, precisions, label=f"PR-AUC: {average_precision_score(y_true, y_score):.3f}")
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].legend()

        # ROC
        axes[2].show_plot(fpr, tpr, linewidth=2, label=f"ROC-AUC: {roc_auc_score(y_true, y_score):.3f}")
        axes[2].show_plot([0, 1], [0, 1], 'k--')
        axes[2].set_xlabel('FPR(=FP/RealNegative)')
        axes[2].set_ylabel('TPR(Recall)')
        axes[2].set_xlim([0, 1])
        axes[2].set_ylim([0, 1])
        axes[2].legend()
