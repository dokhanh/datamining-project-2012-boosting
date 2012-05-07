function trees = LS_Boost(xtrain, ytrain, M)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    N = size(xtrain, 1);
    F = zeros(N, M);
    trees = cell(M, 1);
    for m = 1:M
        if (m > 1)
            ybarre = ytrain - F(:, m-1);
        else
%             ybarre = ytrain - mean(ytrain);
            ybarre = ytrain;
        end
        t = classregtree(xtrain,ybarre,'minparent',N);
        trees{m} = t;
        h = eval(t, xtrain);
        if (m > 1)
            F(:, m) = F(:, m-1) + h;
        else
%             F(:, m) = h + mean(ytrain);
            F(:, m) = h;
        end
    end
end

