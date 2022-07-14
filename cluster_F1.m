function [p, r, f1] = cluster_F1(labels_true, labels_pred)
    labels_true = labels_true(:);
    labels_pred = labels_pred(:);
    [~, fp, fn, tp] = pair_confusion_matrix(labels_true, labels_pred);
    p = tp / (tp + fp);
    r = tp / (tp + fn);
    f1 = 2.0 * (p * r / (p + r));
