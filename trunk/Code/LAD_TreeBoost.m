function trees = LAD_TreeBoost(xtrain, ytrain, M)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

    N = size(xtrain, 1);
    F = zeros(N, M);
    trees = cell(M, 2);
    for m = 1:M
        if (m > 1)
            z = sign(ytrain - F(:, m-1));
        else
            z = sign(ytrain);
        end
        ybarre = sign(z);
        t = classregtree(xtrain,ybarre,'minparent',N);
        trees{m, 1} = t;
        [~, nodes] = eval(t, xtrain);
        values = zeros(1,3);
        values(2) = median(z(nodes == 2));
        values(3) = median(z(nodes == 3));
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

