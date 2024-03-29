function trees = LogitBoost_Multiclass(xtrain, ytrain, M, J)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    N = size(xtrain, 1);
    w = (1/N)*ones(N,J);
    F = zeros(N, J);
    p = (1/J)*ones(N, J);
    trees = cell(M,J);
    ytrainMod = zeros(N, J);
    for j = 1:J
        ytrainMod(:, j) = ytrain == j;
    end
    for m = 1:M
        m
        z = zeros(N, J);
        f = zeros(N, J);
        for j = 1:J
            z(:, j) = (ytrainMod(:, j) - p(:, j))./(p(:, j).*(1-p(:, j)));
            w(:, j) = p(:, j).*(1-p(:, j));
            t = classregtree(xtrain,z(:, j),'minparent',N,'weights',w(:, j));
            trees{m, j} = t;
            f(:, j) = eval(t, xtrain);
        end
        g = sum(f, 2);
        for j = 1:J
            f(:, j) = ((J-1)/J)*(f(:, j) - (1/J)*g);
            F(:, j) = F(:, j) + f(:, j);
        end
        for j = 1:J
            p(:, j) = exp(F(:, j))./(sum(exp(F), 2));
        end
    end
end