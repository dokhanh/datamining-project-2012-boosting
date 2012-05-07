function [resSynt, FSyntExp] = output_AdaBoostMH_imp2(trees, M, xtest, J)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    n = size(xtest, 1);
    d = size(xtest, 2);
    xtestExp = zeros(n*J, d+1);
    xtestExp(:, 1:d) = repmat(xtest, J, 1);
    for j = 1:J
        xtestExp((j-1)*n+1:(j-1)*n+n, d+1) = j;
    end
    [~, FSyntExp] = output_GentleAdaBoost(trees, M, xtestExp);
    resSynt = zeros(n, M);
    for m = 1:M
        for i = 1:n 
            l = FSyntExp(i+(0:J-1)*n, m);
            [~, resSynt(i, m)] = max(l);
        end
    end
end

