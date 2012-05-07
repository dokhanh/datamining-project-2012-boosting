function [resSynt, FSynt] = output_ModestAdaBoost(trees, M, xtest)
%output_ModestAdaBoost The output of the algo Modest AdaBoost with data xtest
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
        [yfitted,nodesFitted] = eval(trees{m, 1}, xtest);
        p = classprob(trees{m, 1},nodesFitted);
        if str2double(yfitted{1}) == -1
            [~,k] = min(p(1,:));
        else
            [~,k] = max(p(1,:));
        end
        p = p(:, k)/2;
        
        %for inverted dist
        [yfitted,nodesFitted] = eval(trees{m, 2}, xtest);
        p_inv = classprob(trees{m, 2}, nodesFitted);
        if str2double(yfitted{1}) == -1
            [~,k] = min(p_inv(1,:));
        else
            [~,k] = max(p_inv(1,:));
        end
        p_inv = p_inv(:, k)/2;
        
        F(:, m) = (p.*(1-p_inv) - (1-p).*p_inv);
    end
    FSynt = zeros(n, M);
    for m = 1:M
        FSynt(:,m) = sum(F(:, 1:m), 2);
    end
    resSynt = 2*(FSynt > 0) - 1;

end