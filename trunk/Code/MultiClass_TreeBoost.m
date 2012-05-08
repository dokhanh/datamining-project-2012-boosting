function trees = MultiClass_TreeBoost(xtrain, ytrain, M, J)
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here

    N = size(xtrain, 1);
    F = zeros(N, J);
    trees = cell(M,J,2);
    ytrainMod = zeros(N, J);
    for j = 1:J
        ytrainMod(:, j) = ytrain == j;
    end
    for m = 1:M
        p = exp(F);
        psum = sum(p, 2);
        for j = 1:J
            p(:, j) = p(:, j)./psum;
        end
        ybarre = ytrainMod - p;
        for j = 1:J
            t = classregtree(xtrain,ybarre(:, j),'minparent',N);
            trees{m, j, 1} = t;
            [~, nodes] = eval(t, xtrain);
            values = zeros(1, 3);
            values(2) = ((J-1)/J)*sum(ybarre(nodes == 2, j))/...
            sum(abs(ybarre(nodes == 2, j)).*(1-abs(ybarre(nodes == 2, j))));
            values(3) = ((J-1)/J)*sum(ybarre(nodes == 3, j))/...
            sum(abs(ybarre(nodes == 3, j)).*(1-abs(ybarre(nodes == 3, j))));
            trees{m, j, 2} = values;
            h = zeros(N, 1);
            for i = 1:N
                h(i) = values(nodes(i));
            end
            F(:, j) = F(:, j) + h;
        end
    end
end

