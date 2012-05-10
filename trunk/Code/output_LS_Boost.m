function [resSynt, FSynt] = output_LS_Boost(trees, M, xtest)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

    n = size(xtest, 1);
    F = zeros(n, M);
    FSynt = zeros(n, M);
    resSynt = zeros(n, M);
    for m = 1:M
        F(:, m) = eval(trees{m}, xtest);
%         FSynt(:, m) = trees{1,2} + sum(F(:, 1:m), 2);
        FSynt(:, m) = sum(F(:, 1:m), 2);
        resSynt(:, m) = 2*(FSynt(:, m) > 0) - 1;
    end
end

