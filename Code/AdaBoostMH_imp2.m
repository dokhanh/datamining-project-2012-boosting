function trees = AdaBoostMH_imp2(xtrain, ytrain, M, J)
%AdaBoostMH_imp2 Another implementation of AdaBoost.MH
%   Detailed explanation goes here

    N = size(xtrain, 1);
    d = size(xtrain, 2);
    xtrainExp = zeros(N*J, d+1);
    xtrainExp(:, 1:d) = repmat(xtrain, J, 1);
    ytrainExp = -1*ones(N*J, 1);
    for j = 1:J
        xtrainExp((j-1)*N+1:(j-1)*N+N, d+1) = j;
        ind = find(ytrain == j);
        ytrainExp((j-1)*N+ind) = 1;
    end
    trees = GentleAdaBoost(xtrainExp, ytrainExp, M);
end

