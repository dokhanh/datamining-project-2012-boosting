function trees = AdaBoostMH(xtrain, ytrain, M, J)
%AdaBoostMH Real AdaBoost for multi-classes
%   Similar to Real AdaBoost

%     N = size(xtrain, 1);
%     d = size(xtrain, 2);
%     xtrainExp = zeros(N*J, d+1);
%     xtrainExp(:, 1:d) = repmat(xtrain, J, 1);
%     ytrainExp = -1*ones(N*J, 1);
%     for j = 1:J
%         xtrainExp((j-1)*N+1:(j-1)*N+N, d+1) = j;
%         ind = find(ytrain == j);
%         ytrainExp((j-1)*N+ind) = 1;
%     end
%     trees = RealAdaBoost(xtrainExp, ytrainExp, M);
    trees = cell(J, M);
    for j = 1:J
        ytrainOAA = 2*(ytrain == j)-1; % one againsts all
        trees(j,:) = RealAdaBoost(xtrain, ytrainOAA, M);
    end
end

