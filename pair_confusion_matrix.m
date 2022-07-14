function [tn, fp, fn, tp] = pair_confusion_matrix(labels_true, labels_pred)
    labels_true = labels_true(:);
    labels_pred = labels_pred(:);
    numSamples = length(labels_true);
    contingency = crosstab(labels_true, labels_pred);
    n_c = sum(contingency, 2);
    n_c = n_c(:);
    n_k = sum(contingency, 1);
    n_k = n_k(:);
    sum_squares = sum(contingency.^2, 'all');
    tp = sum_squares - numSamples;
    fp = sum(contingency * n_k) - sum_squares;
    fn = sum(contingency' * n_c) - sum_squares;
    tn = numSamples^2 - fp - fn - sum_squares;
end