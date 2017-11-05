import numpy as np
from scipy import interp
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc

from logger import float_array_string, get_logger

logger = get_logger(__name__)


def get_roc_curve(y_test, y_score):
    n_classes = y_test.shape[1]

    # Compute ROC curve and ROC area for each class
    roc_curves = {}
    for i in range(n_classes):
        fpr, tpr, threshold = roc_curve(y_test[:, i], y_score[:, i])
        roc_curves[i] = {"fpr": fpr, "tpr": tpr, "threshold": threshold, "roc_auc": auc(fpr, tpr)}

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, threshold = roc_curve(y_test.ravel(), y_score.ravel())
    roc_curves["micro"] = {"fpr": fpr, "tpr": tpr, "threshold": threshold, "roc_auc": auc(fpr, tpr)}

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([roc_curves[i]["fpr"] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, roc_curves[i]["fpr"], roc_curves[i]["tpr"])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    roc_curves["macro"] = {"fpr": all_fpr, "tpr": mean_tpr, "roc_auc": auc(all_fpr, mean_tpr)}

    return roc_curves


def get_precision_recall(y_test, y_score):
    n_classes = y_test.shape[1]

    # Compute PR curve and average precision for each class
    pr_curve = {}
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(y_test[:, i], y_score[:, i])
        precision = np.delete(precision, -1)
        recall = np.delete(recall, -1)
        pr_curve[i] = {"precision": precision, "recall": recall, "thresholds": thresholds,
                       "average_precision": average_precision_score(y_test[:, i], y_score[:, i])}

    # Compute micro-average PR curve and average precision
    precision, recall, thresholds = precision_recall_curve(y_test.ravel(), y_score.ravel())
    precision = np.delete(precision, -1)
    recall = np.delete(recall, -1)
    pr_curve["micro"] = {"precision": precision, "recall": recall, "thresholds": thresholds,
                         "average_precision": average_precision_score(y_test, y_score, "micro")}

    # Compute macro-average PR curve and average precision
    # First aggregate all false positive rates
    all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
    # Then interpolate all PR curves at this points
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        # xp needs to be increasing, but recall is decreasing, hence reverse the arrays
        mean_precision += interp(all_recall, pr_curve[i]["recall"][::-1], pr_curve[i]["precision"][::-1])
    # Finally average it and compute AUC
    mean_precision /= n_classes
    # reverse the arrays back
    all_recall = all_recall[::-1]
    mean_precision = mean_precision[::-1]
    pr_curve["micro"] = {"precision": mean_precision, "recall": all_recall,
                         "average_precision": average_precision_score(y_test, y_score, "macro")}

    # calculate f1 score
    for i in pr_curve:
        precision = pr_curve[i]["precision"]
        recall = pr_curve[i]["recall"]
        pr_curve[i]["f1_score"] = 2 * (precision * recall) / (precision + recall)

    return pr_curve


def get_threshold_by_cutoff(precision, recall, thresholds, cutoff=0.9,
                            cutoff_by="recall", order_by="f1_score"):
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


def get_precision_recall_at_threshold(precision, recall, thresholds, cutoff):
    selected_idx = np.where(thresholds >= cutoff)[0]
    if len(selected_idx) == 0:
        return None, None, None
    else:
        return precision[selected_idx[0]], recall[selected_idx[0]], thresholds[selected_idx[0]]


def evaluate_predictions(y: np.ndarray, pred: np.ndarray):
    relevant = y.sum(axis=0)
    selected = pred.sum(axis=0)
    relevant_selected = (y & pred).sum(axis=0)
    precision = relevant_selected / selected
    recall = relevant_selected / relevant
    f1_score = 2 * (precision * recall) / (precision + recall)
    evaluation_metrics = {
        "relevant": relevant.tolist(),
        "selected": selected.tolist(),
        "percent_selected": pred.mean(axis=0).tolist(),
        "relevant_selected": relevant_selected.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_score": f1_score.tolist()
    }

    if len(pred.shape) == 2:
        y_any = y.any(axis=1)
        pred_any = pred.any(axis=1)
        relevant_any = y_any.sum()
        selected_any = pred_any.sum()
        relevant_selected_any = (y_any & pred_any).sum()
        precision_any = relevant_selected_any / selected_any
        recall_any = relevant_selected_any / relevant_any
        f1_score_any = 2 * (precision_any * recall_any) / (precision_any + recall_any)
        evaluation_metrics.update({
            "relevant_any": relevant_any.tolist(),
            "selected_any": selected_any.tolist(),
            "percent_selected_any": pred_any.mean().tolist(),
            "relevant_selected_any": relevant_selected_any.tolist(),
            "precision_any": precision_any.tolist(),
            "recall_any": recall_any.tolist(),
            "f1_score_any": f1_score_any.tolist()
        })

    return evaluation_metrics


def log_evaluation(evaluation_metrics: dict):
    if "labels" in evaluation_metrics:
        logger.info("labels: \t%s.", evaluation_metrics["labels"])
    if "threshold" in evaluation_metrics:
        logger.info("threshold: \t%s.", float_array_string(evaluation_metrics["threshold"]))

    for metric in ["relevant", "selected", "relevant_selected"]:
        if metric in evaluation_metrics:
            logger.info("%s: \t%s.", metric, evaluation_metrics[metric])
    for metric in ["precision", "recall", "percent_selected"]:
        if metric in evaluation_metrics:
            logger.info("$s: \t%s.", metric, float_array_string(evaluation_metrics[metric]))

    logger.info("aggregated any, relevant:  %s, selected:  %s, relevant & selected:  %s, "
                "precision  %.4f, recall:  %.4f, percent selected:  %.4f.",
                evaluation_metrics["relevant_any"], evaluation_metrics["selected_any"],
                evaluation_metrics["relevant_selected_any"],
                evaluation_metrics["precision_any"], evaluation_metrics["recall_any"],
                evaluation_metrics["percent_selected_any"])


def log_predictions(pred: np.ndarray):
    logger.info("prediction shape: %s.", pred.shape)
    logger.info("selected: \t%s.", pred.sum(axis=0).tolist())
    logger.info("percent selected: \t%s.", [float("{:.4f}".format(el)) for el in pred.mean(axis=0)])
    if len(pred.shape) == 2:
        logger.info("selected any:  %s.", pred.any(axis=1).sum())
        logger.info("percent selected any:  %s.", float("{:.4f}".format(pred.any(axis=1).mean())))
