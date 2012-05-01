function regres = LogitBoost_2class(xtrain, ytrain, M)
%LogitBoost Logit Boost Algorithm
%   return a set of weighted least-squares regressions

    N = size(xtrain, 1);
    w = (1/N)*ones(N,1);
    F = zeros(N, 1);
    p = (1/2)*ones(N, 1);
    regres = cell(M,1);
end

