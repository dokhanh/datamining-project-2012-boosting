function [trees, err, C] = DiscreteAdaBoost(xtrain, ytrain, M)
%DiscreteAdaBoost Boosting Algorithm
%   return a set of regression trees in 'trees', errors in 'err', and the
%   coefficients for each of these base classifiers in 'C'

    N = size(xtrain, 1);
    w = (1/N)*ones(N,1);
    trees = cell(M,1);
    err = zeros(M,1); % error at each iteration
    C = zeros(M,1);
    for m = 1:M
        m
        t = classregtree(xtrain,ytrain,'minparent',N,'weights',w,'method','classification');
        trees{m} = t;
        ytrained = eval(t,xtrain);
        err(m) = 0;
        for i = 1:N
            err(m) = err(m)+(str2double(ytrained{i}) ~= ytrain(i))*w(i);
        end
    %     err(m) = cval(5, xtrain, ytrain, w);
        if (err(m) < 0.5)
            C(m) = log((1 - err(m))/err(m));
        end
        for i = 1:N
            w(i) = w(i)*exp(C(m)*(str2double(ytrained{i}) ~= ytrain(i)));
        end
        w = w/sum(w);
    end
end

