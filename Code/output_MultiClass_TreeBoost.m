function [resSynt, FSynt] = output_MultiClass_TreeBoost(trees, M, xtest, J)
%UNTITLED12 Summary of this function goes here
%   Detailed explanation goes here

    n = size(xtest, 1);
    F = zeros(M, J, n);
    FSynt = zeros(M, J, n);
    resSynt = zeros(n, M);
    for m = 1:M
        for j = 1:J
            [~, nodes] = eval(trees{m, j, 1}, xtest);
            values = trees{m, j, 2};
            for i = 1:n
                F(m, j, i) = values(nodes(i));
            end
        end
    end
    for i = 1:n
        for m = 1:M
            FSynt(m, :, i) = sum(F(1:m, :, i));
            [~, resSynt(i, m)] = max(FSynt(m, :, i));
        end
    end
end

