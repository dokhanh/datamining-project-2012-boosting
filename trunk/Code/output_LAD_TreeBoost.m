function [resSynt, FSynt] = output_LAD_TreeBoost(trees, M, xtest)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

    n = size(xtest, 1);
    F = zeros(n, M);
    FSynt = zeros(n, M);
    resSynt = zeros(n, M);
    for m = 1:M
        %F(:, m) = eval(trees{m}, xtest);
        [~, nodes] = eval(trees{m, 1}, xtest);
        values = trees{m, 2};
        for i = 1:n
            F(i, m) = values(nodes(i));
        end
        FSynt(:, m) = sum(F(:, 1:m), 2);
        resSynt(:, m) = 2*(FSynt(:, m) > 0) - 1;
    end
end

