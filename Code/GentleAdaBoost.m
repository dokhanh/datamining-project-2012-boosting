function [trees err] = GentleAdaBoost(xtrain, ytrain, M)
%   Gentle AdaBoost execute the Gentle Adaboost Algorithm [Friedman et al. 1988]
%   Input:
%       xtrain, ytrain: training data set
%       M:  number of iteration
%   Output:
%       trees:  a cell containing base classifiers
%       err:    training error

    N = size(xtrain, 1);
    w = (1/N)*ones(N,1);
    trees = cell(M,1);
    err = 0;
    
    for m = 1:M
        m
        t = classregtree(xtrain,ytrain,'minparent',N,'weights',w,'method','classification');
        trees{m} = t;
        [ytrained,nodesTrained] = eval(t, xtrain);
        p = classprob(t,nodesTrained);
        if str2double(ytrained{1}) == -1
            [~,k] = min(p(1,:));
        else
            [~,k] = max(p(1,:));
        end
        p = p(:, k);
        f = 2*p - 1;
        w = w.*exp(-ytrain.*f);
        w = w/sum(w);
    end
end