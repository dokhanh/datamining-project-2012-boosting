function [resSynt, F] = output_AdaBoostMH(trees, M, xtest, J, type)
%output_AdaBoostMH
%   

    n = size(xtest, 1);
    F = zeros(n, M, J);
    R = zeros(n, M, J);
    for j = 1:J
        switch type
            case 1
                [R(:,:,j), F(:,:,j)] = output_GentleAdaBoost(trees(:, j), M, xtest);
            case 2
                [R(:,:,j), F(:,:,j)] = output_LS_Boost(trees(:, j), M, xtest);
            case 3
                [R(:,:,j), F(:,:,j)] = output_LAD_TreeBoost(trees(:, 2*j-1:2*j), M, xtest);
            case 4
                [R(:,:,j), F(:,:,j)] = output_M_TreeBoost(trees(:, 3*j-2:3*j), M, xtest);
            case 5
                [R(:,:,j), F(:,:,j)] = output_LogitBoost_2class(trees(:, j), M, xtest);
            otherwise
                [R(:,:,j), F(:,:,j)] = output_L2_TreeBoost(trees(:, 2*j-1:2*j), M, xtest);
%         [R(:,:,j), F(:,:,j)] = output_DiscreteAdaBoost(trees(j,:), C, M, xtest);
        end
    end
    resSynt = zeros(n, M);
    for m = 1:M
        for i = 1:n 
            l = R(i,m,:);
            [~, resSynt(i, m)] = max(F(i,m,:));
        end
    end
end

