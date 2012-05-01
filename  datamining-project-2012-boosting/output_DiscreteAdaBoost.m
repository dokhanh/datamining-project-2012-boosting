function resSynt = output_DiscreteAdaBoost(trees, C, M, xtest)
%output_DiscreteAdaBoost retourne les classifications pour Discrete
%AdaBoost
%   Input:
%   trees: classifieurs de base
%   C: coefficients pour classifieurs de base
%   M: nombre d'itérations = nombre de classifieurs de base
%   xtest: input data
%   Output:
%   resSynt: les résultats après chaque itération
    
    n = size(xtest, 1);
    res = zeros(n, M);
    for m = 1:M
        l = eval(trees{m}, xtest);
        for i = 1:n
            res(i, m) = str2double(l{i});
        end
    end
    resSynt = zeros(n, M);
    for m = 1:M
        resSynt(:, m) = 2*(res(:, 1:m)*C(1:m, 1) > 0) - 1;
    end
end

