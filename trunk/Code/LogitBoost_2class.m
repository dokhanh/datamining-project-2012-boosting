function trees = LogitBoost_2class(xtrain, ytrain, M)
%LogitBoost Logit Boost Algorithm
%   return a set of weighted least-squares regressions

    N = size(xtrain, 1);
    w = (1/N)*ones(N,1);
    F = zeros(N, 1);
    p = (1/2)*ones(N, 1);
    trees = cell(M,1);
    ytrainMod = (ytrain+1)/2;
    for m = 1:M
        m
        z = (ytrainMod - p)./(p.*(1-p));
        w = p.*(1-p);
        t = classregtree(xtrain,z,'minparent',N,'weights',w);
        trees{m} = t;
        f = eval(t, xtrain);
        F = F + (1/2)*f;
        p = exp(F)./(exp(F) + exp(-F));
    end
end

