function [resSynt, F] = output_DiscreteAdaBoost(trees, C, M, xtest)
%output_DiscreteAdaBoost retourne les classifications pour Discrete
%AdaBoost
%   Input:
%   trees: classifieurs de base
%   C: coefficients pour classifieurs de base
%   M: nombre d'it�rations = nombre de classifieurs de base
%   xtest: input data
%   Output:
%   resSynt: les r�sultats apr�s chaque it�ration
    
    n = size(xtest, 1);
    res = zeros(n, M);
    for m = 1:M
        l = eval(trees{m}, xtest);
        for i = 1:n
            res(i, m) = str2double(l{i});
        end
    end
    resSynt = zeros(n, M);
    F = zeros(n, M);
    for m = 1:M
        F(:, m) = res(:, 1:m)*C(1:m, 1);
        resSynt(:, m) = 2*(F(:, m) > 0) - 1;
    end
end

