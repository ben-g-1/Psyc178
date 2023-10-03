function glm_graul(X, Y, C)
%   Manual GLM Script for Psyc 178
%   By Ben Graul
%   v0.2 9/27/23


    % Check if a contrast matrix is provided
    if nargin < 3
        C = [];
    end

    [n, p] = size(X); % record rows and columns

    nans = isnan(X); % record NaNs
    X = rmmissing(X); % remove NaNs

    gram = X' * X; % Gram matrix

    invgram = gram^-1;

    err_freedom = n - p; % requires n > p, or number of observations > parameters

    % Beta
    b_coeff = invgram * X' * Y; % beta hat

    % Residuals
    res = Y - X * b_coeff;

    var_res = res' * res ./ err_freedom * invgram;
    var_sq = res' * res ./ err_freedom;

    % Variance of Beta
    var_coeff = var_res^2 * invgram;

    % Standard Error of Coefficients
    SE_coeff = sqrt(diag(var_coeff));

    % Calculate the sum of squared residuals
    sumsqerr = res' * res;

    % Calculate the total sum of squares
    sumsqtotal = (Y - mean(Y))' * (Y - mean(Y));

    % Coefficient of Determination (R-squared)
    r2 = 1 - sumsqerr / sumsqtotal;

    % Calculate t-statistics and p-values for each coefficient
    tstat = b_coeff ./ SE_coeff;
    pval = (1 - tcdf(abs(tstat), err_freedom)) * 2;

    % Display t-values, p-values, and degrees of freedom
    fprintf('Coefficient   t-value   P-value   Degrees of Freedom\n');
    for i = 1:p
        fprintf('%d             %.4f    %.4f    %d\n', i, tstat(i), pval(i), err_freedom);
    end

    if ~isempty(C)
        % Perform t-tests for contrasts
        Chat = C' * b_coeff;
        var_Chat = var_sq * C' * invgram * C;
        SE_Chat = sqrt(diag(var_Chat));
        tstat_Chat = Chat ./ SE_Chat;
        pval_Chat = (1 - tcdf(abs(tstat_Chat), err_freedom)) * 2;

        % Display contrast results
        fprintf('\nContrast Results:\n');
        for i = 1:size(C, 2)
            fprintf('Contrast %d   t-value   P-value   Degrees of Freedom\n', i);
            fprintf('            %.4f    %.4f    %d\n', tstat_Chat(i), pval_Chat(i), err_freedom);
        end
    end

    % Testing Full Model Against Intercept-Only Model
    X_intercept = ones(n, 1);
    b_coeff_intercept = (X_intercept' * X_intercept)^(-1) * X_intercept' * Y;

    % Calculate the sum of squared residuals for the intercept-only model
    res_intercept = Y - X_intercept * b_coeff_intercept;
    sumsqerr_intercept = res_intercept' * res_intercept;

    % Calculate degrees of freedom for both models
    p_full = p + 1; % Full model includes the intercept
    p_intercept = 1; % Intercept-only model has 1 parameter

    % Calculate the F-statistic
    F_statistic = ((sumsqerr_intercept - sumsqerr) / (p_full - p_intercept)) / (sumsqerr / (n - p_full));
    
    % Calculate the p-value for the F-statistic using the F-distribution
    p_value_F = 1 - fcdf(F_statistic, p_full - p_intercept, n - p_full);

    % Display F-statistic and p-value
    fprintf('\nOverall Model (Full vs. Intercept-Only):\n');
    fprintf('F-statistic: %.4f\n', F_statistic);
    fprintf('p-value for F-statistic: %.4f\n', p_value_F);
end
