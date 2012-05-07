function trees = AdaBoostMH(xtrain, ytrain, M, J)
%AdaBoostMH Real AdaBoost for multi-classes
%   Similar to Real AdaBoost

    trees = cell(J, M);
    C = zeros(M, J);
    for j = 1:J
        ytrainOAA = 2*(ytrain == j)-1; % one againsts all
        trees(j,:) = LogitBoost_2class(xtrain, ytrainOAA, M);
%         [trees(j, :), ~, C(:, j)] = DiscreteAdaBoost(xtrain, ytrain, M);
    end
end

