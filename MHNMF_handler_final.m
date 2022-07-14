clear all;
dataset = 'wisconsin';
model = 'MHNMF';
fprintf('dataset is %s.\n', dataset);
X = load(strcat(dataset, '_A.mat')).A;
tempX = speye(size(X));
AdjCell = cell(1, 6);
%% =================construct adjacency matrices=======================
for k = 1:6
    tempX = tempX * X; % $X^k$.
    AdjCell{k} = NormAdjac(tempX);
end

clear tempX;
%% ==========================constructed===============================
target = load(strcat(dataset, '_target.mat')).T;
k = 10;
paralist = [1e-3 1e-2 1e-1 1 1e1];
dup = linspace(1, 10, 10);
[para1, para2, tau] = ndgrid(paralist, paralist, dup);
purity_max = zeros(numel(tau), 1);
pre_max = zeros(numel(tau), 1);
rec_max = zeros(numel(tau), 1);
f1_max = zeros(numel(tau), 1);
comSubsets = cell(numel(tau), 1);

parfor ind_grid = 1:numel(tau)
    rng(ind_grid);
    [U, ~] = MHNMF(AdjCell, k, para1(ind_grid), para2(ind_grid), false);
    [~, q] = max(U, [], 2);
    comSubsets{ind_grid, 1} = q;
    purity = cluster_Purity_aba(target, q);
    [p, r, f1] = cluster_F1(target, q);
    purity_max(ind_grid) = purity;
    pre_max(ind_grid) = p;
    rec_max(ind_grid) = r;
    f1_max(ind_grid) = f1;
    fprintf('This is %d-th search (total %d.). Purity is %f.\n', ind_grid, numel(tau), purity);
end

fprintf('Dataset is %s.\n', dataset);

reshape_purity = mean(reshape(purity_max, [], 10), 2);
disp(max(reshape_purity));
reshape_f1 = mean(reshape(f1_max, [], 10), 2);
disp(max(reshape_f1));