function [resSynt, FSynt] = output_RealAdaBoost(trees, M, xtest)
%output_RealAdaBoost The output of the algo Real AdaBoost with data xtest
%   Input:
%   trees: set of base classifiers obtained by RealAdaBoost
%   M: number of iterations
%   xtest: testing dataset
%   Output:
%   FSynt: real f
%   resSynt: sign of f, take values in {-1, 1}

    n = size(xtest, 1);
    F = zeros(n, M);
    for m = 1:M
        [yfitted,nodesFitted] = eval(trees{m}, xtest);
        p = classprob(trees{m},nodesFitted);
        if str2double(yfitted{1}) == -1
            [~,k] = min(p(1,:));
        else
            [~,k] = max(p(1,:));
        end
        p = p(:, k);
        p = max(p, 0.0000001);
        p = min(p, 1-0.0000001);
        F(:, m) = (1/2)*log(p./(1-p));
    end
    FSynt = zeros(n, M);
    for m = 1:M
        FSynt(:,m) = sum(F(:, 1:m), 2);
    end
    resSynt = 2*(FSynt > 0) - 1;
end

