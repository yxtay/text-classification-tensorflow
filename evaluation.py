import numpy as np
from scipy import interp
from sklearn import metrics

from logger import float_array_string, get_logger, get_name

name = get_name(__name__, __file__)
logger = get_logger(name)


def get_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    calculates roc curve data from y true and prediction scores
    includes fpr, tpr, thresholds, roc_auc
    at each level of y, micro and macro averaged

    Args:
        y_true: true y values
        y_score: y prediction scores

    Returns:
        dict with roc curve data
    """
    n_classes = y_true.shape[1]

    # Compute ROC curve and ROC area for each class
    roc_curves = {}
    for i in range(n_classes):
        fpr, tpr, thresholds = metrics.roc_curve(y_true[:, i], y_score[:, i])
        roc_curves[i] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "roc_auc": metrics.auc(fpr, tpr)}

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, thresholds = metrics.roc_curve(y_true.ravel(), y_score.ravel())
    roc_curves["micro"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "roc_auc": metrics.auc(fpr, tpr)}

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([roc_curves[i]["fpr"] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, roc_curves[i]["fpr"], roc_curves[i]["tpr"])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    roc_curves["macro"] = {"fpr": all_fpr, "tpr": mean_tpr, "roc_auc": metrics.auc(all_fpr, mean_tpr)}

    return roc_curves


def get_precision_recall(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    calculates precision-recall curve data from y true and prediction scores
    includes precision, recall, f1_score, thresholds, average_precision
    at each level of y, micro and macro averaged

    Args:
        y_true: true y values
        y_score: y prediction scores

    Returns:
        dict with precision-recall curve data
    """
    n_classes = y_true.shape[1]

    # Compute PR curve and average precision for each class
    pr_curves = {}
    for i in range(n_classes):
        precision, recall, thresholds = metrics.precision_recall_curve(y_true[:, i], y_score[:, i])
        precision = np.delete(precision, -1)
        recall = np.delete(recall, -1)
        pr_curves[i] = {"precision": precision, "recall": recall, "thresholds": thresholds,
                        "average_precision": metrics.average_precision_score(y_true[:, i], y_score[:, i])}

    # Compute micro-average PR curve and average precision
    precision, recall, thresholds = metrics.precision_recall_curve(y_true.ravel(), y_score.ravel())
    precision = np.delete(precision, -1)
    recall = np.delete(recall, -1)
    pr_curves["micro"] = {"precision": precision, "recall": recall, "thresholds": thresholds,
                          "average_precision": metrics.average_precision_score(y_true, y_score, "micro")}

    # Compute macro-average PR curve and average precision
    # First aggregate all false positive rates
    all_recall = np.unique(np.concatenate([pr_curves[i]["recall"] for i in range(n_classes)]))
    # Then interpolate all PR curves at this points
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        # xp needs to be increasing, but recall is decreasing, hence reverse the arrays
        mean_precision += interp(all_recall, pr_curves[i]["recall"][::-1], pr_curves[i]["precision"][::-1])
    # Finally average it and compute AUC
    mean_precision /= n_classes
    # reverse the arrays back
    all_recall = all_recall[::-1]
    mean_precision = mean_precision[::-1]
    pr_curves["micro"] = {"precision": mean_precision, "recall": all_recall,
                          "average_precision": metrics.average_precision_score(y_true, y_score, "macro")}

    # calculate f1 score
    for i in pr_curves:
        precision = pr_curves[i]["precision"]
        recall = pr_curves[i]["recall"]
        pr_curves[i]["f1_score"] = 2 * (precision * recall) / (precision + recall)

    return pr_curves


def get_threshold_by_cutoff(precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray,
                            cutoff: float = 0.9, cutoff_by: str = "recall", order_by: str = "f1_score"
                            ) -> (float, float, float):
    """
    calculates precision, recall and threshold
    at specified metric cutoff

    Args:
        precision: precision curve
        recall: recall curve
        thresholds: associated thresholds
        cutoff: cutoff value
        cutoff_by: metric to apply cutoff
        order_by: metric to order by

    Returns:
        precision, recall and threshold
    """
    f1_score = 2 * (recall * precision) / (precision + recall)
    scores = {"precision": precision, "recall": recall, "f1_score": f1_score}
    if not (scores[cutoff_by] >= cutoff).any():
        return None, None, None
    else:
        # get indexes sorted by f1 score without NAs
        sorted_idx = scores[order_by].argsort()[::-1]
        sorted_idx = sorted_idx[~np.isnan(f1_score[sorted_idx])]
        selected_idx = sorted_idx[np.where(scores[cutoff_by][sorted_idx] >= cutoff)[0]]
        return precision[selected_idx[0]], recall[selected_idx[0]], thresholds[selected_idx[0]]


def get_precision_recall_at_threshold(precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray,
                                      cutoff: float) -> (float, float, float):
    """
    calculates precision, recall and threshold
    at specified threshold cutoff

    Args:
        precision: precision curve
        recall: recall curve
        thresholds: associated thresholds
        cutoff: threshold cutoff value

    Returns:
        precision, recall and threshold
    """
    selected_idx = np.where(thresholds >= cutoff)[0]
    if len(selected_idx) == 0:
        return None, None, None
    else:
        return precision[selected_idx[0]], recall[selected_idx[0]], thresholds[selected_idx[0]]


def evaluate_prediction_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    calculates evaluation metrics from y true and prediction scores
    includes log_loss, roc_auc and average_precision

    Args:
        y_true: true y values
        y_score: y prediction scores

    Returns:
        dictionary with evaluation metrics
    """
    # sklearn metrics
    log_loss = metrics.log_loss(y_true, y_score)
    macro_roc_auc = metrics.roc_auc_score(y_true, y_score, average="macro")
    micro_roc_auc = metrics.roc_auc_score(y_true, y_score, average="micro")
    macro_average_precision = metrics.average_precision_score(y_true, y_score, average="macro")
    micro_average_precision = metrics.average_precision_score(y_true, y_score, average="micro")

    evaluation = {
        "log_loss": log_loss,
        "roc_auc": macro_roc_auc,
        "macro_roc_auc": macro_roc_auc,
        "micro_roc_auc": micro_roc_auc,
        "average_precision": macro_average_precision,
        "macro_average_precision": macro_average_precision,
        "micro_average_precision": micro_average_precision,
    }
    return evaluation


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
        calculates evaluation metrics from y true and prediction
        includes relevant, selected, relevant_selected counts,
        precision, recall, f1_score, percent_selected,
        accuracy, jaccard, hamming, zero_one and report

        Args:
            y_true: true y values
            y_pred: y prediction

        Returns:
            dictionary with evaluation metrics
    """
    # precision & recall
    relevant = y_true.sum(axis=0)
    selected = y_pred.sum(axis=0)
    relevant_selected = (y_true & y_pred).sum(axis=0)
    precision = relevant_selected / selected
    recall = relevant_selected / relevant
    f1_score = 2 * (precision * recall) / (precision + recall)

    # sklearn metrics
    accuracy = metrics.accuracy_score(y_true, y_pred)
    jaccard = metrics.jaccard_similarity_score(y_true, y_pred)
    hamming = metrics.hamming_loss(y_true, y_pred)
    zero_one = metrics.zero_one_loss(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred)

    evaluation = {
        "relevant": relevant.tolist(),
        "selected": selected.tolist(),
        "percent_selected": y_pred.mean(axis=0).tolist(),
        "relevant_selected": relevant_selected.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_score": f1_score.tolist(),
        "accuracy": accuracy,
        "jaccard": jaccard,
        "hamming": hamming,
        "zero_one": zero_one,
        "report": report,
    }

    if len(y_pred.shape) == 2:
        y_true_any = y_true.any(axis=1)
        y_pred_any = y_pred.any(axis=1)

        # precision & recall
        relevant_any = y_true_any.sum()
        selected_any = y_pred_any.sum()
        relevant_selected_any = (y_true_any & y_pred_any).sum()
        precision_any = relevant_selected_any / selected_any
        recall_any = relevant_selected_any / relevant_any
        f1_score_any = 2 * (precision_any * recall_any) / (precision_any + recall_any)

        # sklearn metrics
        accuracy_any = metrics.accuracy_score(y_true_any, y_pred_any)
        jaccard_any = metrics.jaccard_similarity_score(y_true_any, y_pred_any)
        hamming_any = metrics.hamming_loss(y_true_any, y_pred_any)
        zero_one_any = metrics.zero_one_loss(y_true_any, y_pred_any)
        report_any = metrics.classification_report(y_true_any, y_pred_any)

        evaluation.update({
            "relevant_any": relevant_any.tolist(),
            "selected_any": selected_any.tolist(),
            "percent_selected_any": y_pred_any.mean().tolist(),
            "relevant_selected_any": relevant_selected_any.tolist(),
            "precision_any": precision_any.tolist(),
            "recall_any": recall_any.tolist(),
            "f1_score_any": f1_score_any.tolist(),
            "accuracy_any": accuracy_any,
            "jaccard_any": jaccard_any,
            "hamming_any": hamming_any,
            "zero_one_any": zero_one_any,
            "report_any": report_any,
        })

    return evaluation


def log_evaluation(evaluation: dict) -> None:
    """
    logs evaluation metrics of predictions

    Args:
        evaluation: dictionary from `evaluate_prediction()`
    """
    if "labels" in evaluation:
        logger.info("labels: \t%s.", evaluation["labels"])
    if "threshold" in evaluation:
        logger.info("threshold: \t%s.", float_array_string(evaluation["threshold"]))

    for metric in ["relevant", "selected", "relevant_selected"]:
        if metric in evaluation:
            logger.info("%s: \t%s.", metric, evaluation[metric])
    for metric in ["precision", "recall", "f1_score", "percent_selected"]:
        if metric in evaluation:
            logger.info("%s: \t%s.", metric, float_array_string(evaluation[metric]))

    logger.info("aggregated any, relevant:  %s, selected:  %s, relevant & selected:  %s, "
                "precision  %.4f, recall:  %.4f, f1_score_any:  %.4f, percent selected:  %.4f.",
                *[evaluation[metric] for metric in
                  ["relevant_any", "selected_any", "relevant_selected_any",
                   "precision_any", "recall_any", "f1_score_any",
                   "percent_selected_any"]])


def log_predictions(y_pred: np.ndarray) -> None:
    """
    logs predictions
    includes selected and percent selected

    Args:
        y_pred: y prediction
    """
    logger.info("labels shape: %s.", y_pred.shape)
    logger.info("selected: \t%s.", y_pred.sum(axis=0).tolist())
    logger.info("percent selected: \t%s.", float_array_string(y_pred.mean(axis=0)))
    logger.info("selected any:  %s.", y_pred.any(axis=1).sum())
    logger.info("percent selected any:  %.4f.", y_pred.any(axis=1).mean())
