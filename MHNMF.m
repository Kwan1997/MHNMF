function [H_star, H] = MHNMF(AdjCell, k, lambda, eta, verbose)
    % get the number of different hops.
    m = length(AdjCell);
    % get the number of nodes.
    n = size(AdjCell{1}, 1);
    % initialize community assigment.
    H = cell(1, m);

    for ind = 1:m
        H{ind} = rand(n, k);
    end

    H_star = rand(n, k);
    last_Hstar = H_star;
    % initial view weights.
    alfa = ones(m, 1) ./ m;
    mju = ones(m, 1) ./ m;
    % set epsilon, and max iterations.
    xi = eps;
    tol = 1e-4;
    iter = 0;
    maxIter = 200;
    % start iterative optimization.
    while iter < maxIter
        % Update H^{v}.
        for ind = 1:m
            HHH = H{ind} * H{ind}' * H{ind};
            Delta = (-2 .* alfa(ind) .* H{ind}) .* (-2 .* alfa(ind) .* H{ind}) + 16 .* HHH .* (4 .* AdjCell{ind} * H{ind} + 2 .* alfa(ind) .* H_star);
            H{ind} = H{ind} .* abs(sqrt((-2 .* alfa(ind) .* H{ind} + abs(sqrt(Delta))) ./ max(xi, 8 .* HHH)));
        end

        % Update H^{*}.
        numerator = 0;

        for ind = 1:m
            numerator = numerator + alfa(ind) .* H{ind} + lambda .* mju(ind) .* AdjCell{ind} * H_star;
        end

        H_star = H_star .* abs(sqrt(numerator ./ max(xi, (1 + lambda) .* H_star)));
        % Update \alpha.
        zeda = zeros(m, 1);

        for ind = 1:m
            zeda(ind) = norm(H_star - H{ind}, 'fro').^2;
        end

        alfa = projsplx(-zeda ./ (2.0 .* eta));
        % Update \mu.
        zeda2 = zeros(m, 1);

        for ind = 1:m
            zeda2(ind) = trace(H_star' * (speye(n) - AdjCell{ind}) * H_star);
        end

        mju = projsplx((-lambda .* zeda2) ./ (2.0 .* eta));
        % Report optimization process.
        if verbose
            obj = 0;
            cons_L = zeros(n, n);

            for ind = 1:m
                obj = obj + norm(AdjCell{ind} - H{ind} * H{ind}', 'fro').^2 + alfa(ind) .* norm(H_star - H{ind}, 'fro').^2;
                cons_L = cons_L + mju(ind) .* (speye(n) - AdjCell{ind});
            end

            obj = obj + eta .* norm(alfa).^2 + eta .* norm(mju).^2 + lambda .* trace(H_star' * cons_L * H_star);
            fprintf('This is %d th iteration, loss = %f.\n', iter + 1, obj);
        end

        % Check convergence
        if max(abs(H_star - last_Hstar), [], 'all') <= tol
            break;
        end

        last_Hstar = H_star;

        % Update iteration number.
        iter = iter + 1;
    end

end
