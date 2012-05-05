function [resSynt, FSynt] = output_LogitBoost_2class(trees, M, xtest)
%output_LogitBoost_2class output of Logit Boost 2-class version
%   The same as other output functions

    n = size(xtest, 1);
    F = zeros(n, M);
    for m = 1:M
        F(:, m) = eval(trees{m}, xtest);
    end
    FSynt = zeros(n, M);
    for m = 1:M
        FSynt(:,m) = sum(F(:, 1:m), 2);
    end
    resSynt = 2*(FSynt > 0) - 1;
end

