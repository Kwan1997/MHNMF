function [purity] = cluster_Purity(labels_true, labels_pred)
    labels_true = labels_true(:)';
    labels_pred = labels_pred(:)';
    total_purity = 0;

    for i1 = unique(labels_pred)
        [~, F] = mode(labels_true(find(labels_pred == i1)));
        total_purity = total_purity + (F / sum(labels_pred == i1));
    end

    purity = total_purity / length(unique(labels_pred));
end
