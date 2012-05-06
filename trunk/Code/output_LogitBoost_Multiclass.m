function [resSynt, FSynt] = output_LogitBoost_Multiclass(trees, M, xtest, J)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    n = size(xtest, 1);
    F = zeros(M, J, n);
    FSynt = zeros(M, J, n);
    resSynt = zeros(n, M);
    for m = 1:M
        for j = 1:J
            F(m, j, :) = eval(trees{m, j}, xtest);
        end
    end
    for i = 1:n
        g = sum(F(:, :, i), 2);
        for j = 1:J
            F(:, j, i) = ((J-1)/J)*(F(:, j, i) - (1/J)*g);
        end
        for m = 1:M
            FSynt(m, :, i) = sum(F(1:m, :, i));
            [~, resSynt(i, m)] = max(FSynt(m, :, i));
        end
    end
end

