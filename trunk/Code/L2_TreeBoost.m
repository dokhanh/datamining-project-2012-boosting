function trees = L2_TreeBoost(xtrain, ytrain, M)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

    N = size(xtrain, 1);
    F = zeros(N, M);
    trees = cell(M, 2);
%     trees{1, 3} = (1/2)*log((1+ytrain)./(1-ytrain));
    for m = 1:M
        if (m > 1)
            ybarre = 2*ytrain./(1+exp(2*ytrain.*F(:, m-1)));
        else
            ybarre = ytrain;
        end
        t = classregtree(xtrain,ybarre,'minparent',N);
        trees{m, 1} = t;
        [~, nodes] = eval(t, xtrain);
        values = zeros(1,3);
        values(2) = sum(ybarre(nodes == 2))/(sum(abs(ybarre(nodes == 2)).*(2 - abs(ybarre(nodes == 2)))));
        values(3) = sum(ybarre(nodes == 3))/(sum(abs(ybarre(nodes == 3)).*(2 - abs(ybarre(nodes == 3)))));
        trees{m, 2} = values;
        h = zeros(N, 1);
        for i = 1:N
            h(i) = values(nodes(i));
        end
        if (m > 1)
            F(:, m) = F(:, m-1) + h;
        else
            F(:, m) = h;
        end
    end
end

