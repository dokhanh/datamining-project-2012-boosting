function trees = AdaBoostMH(xtrain, ytrain, M, J, type)
%AdaBoostMH Real AdaBoost for multi-classes
%   Similar to Real AdaBoost

    trees = cell(M, 3*J);
    C = zeros(M, J);
    for j = 1:J
        ytrainOAA = 2*(ytrain == j)-1; % one againsts all
        switch type
            case 1
                trees(:, j) = GentleAdaBoost(xtrain, ytrainOAA, M);
            case 2
                trees(:, j) = LS_Boost(xtrain, ytrainOAA, M);
            case 3
                trees(:, 2*j-1:2*j) = LAD_TreeBoost(xtrain, ytrainOAA, M);
            case 4
                trees(:, 3*j-2:3*j) = M_TreeBoost(xtrain, ytrainOAA, M);
            case 5
                trees(:, j) = LogitBoost_2class(xtrain, ytrainOAA, M);
            otherwise
                trees(:, 2*j-1:2*j) = L2_TreeBoost(xtrain, ytrainOAA, M);
        end
%         [trees(j, :), ~, C(:, j)] = DiscreteAdaBoost(xtrain, ytrain, M);
    end
end

