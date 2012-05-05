function [resSynt, F, R] = output_AdaBoostMH(trees, M, xtest, J)
%output_AdaBoostMH
%   

    n = size(xtest, 1);
%     d = size(xtest, 2);
%     xtestExp = zeros(n*J, d+1);
%     xtestExp(:, 1:d) = repmat(xtest, J, 1);
%     for j = 1:J
%         xtestExp((j-1)*n+1:(j-1)*n+n, d+1) = j;
%     end
    F = zeros(n, M, J);
    R = zeros(n, M, J);
    for j = 1:J
        [R(:,:,j), F(:,:,j)] = output_RealAdaBoost(trees(j,:), M, xtest);
    end
%     [~, FSyntExp] = output_RealAdaBoost(trees, M, xtestExp);
    resSynt = zeros(n, M);
    for m = 1:M
        for i = 1:n 
            l = R(i,m,:);
%             l = FSyntExp(i+(0:J-1)*n, m);
%             [~, resSynt(i, m)] = max(l);
            [~, resSynt(i, m)] = max(F(i,m,:));
        end
    end
end

