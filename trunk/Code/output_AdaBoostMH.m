function [resSynt, F] = output_AdaBoostMH(trees, M, xtest, J)
%output_AdaBoostMH
%   

    n = size(xtest, 1);
    F = zeros(n, M, J);
    R = zeros(n, M, J);
    for j = 1:J
        [R(:,:,j), F(:,:,j)] = output_LogitBoost_2class(trees(j,:), M, xtest);
%         [R(:,:,j), F(:,:,j)] = output_DiscreteAdaBoost(trees(j,:), C, M, xtest);
    end
    resSynt = zeros(n, M);
    for m = 1:M
        for i = 1:n 
            l = R(i,m,:);
            [~, resSynt(i, m)] = max(F(i,m,:));
        end
    end
end

