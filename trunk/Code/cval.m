function err = cval(K, xtrain, ytrain, w)
%cval calculer erreur par cross validation
%   K: number of folds
%   w: weights of observations

    n = size(xtrain, 1);
    xtrain1 = []; ytrain1 = [];
    nm = n*10*w;
    for i = 1:n
        s = round(nm(i));
        if (s > 0)
            xtrain1 = [xtrain1; repmat(xtrain(i, :), s, 1)];
            ytrain1 = [ytrain1; repmat(ytrain(i, :), s, 1)];
        end
    end
    xtrain = xtrain1;
    ytrain = ytrain1;
    n = size(xtrain, 1);
    indF = unidrnd(K, n, 1);
    res = zeros(n, 1);
    for k = 1:K
        xtrainF = xtrain(indF ~= k, :);
        ytrainF = ytrain(indF ~= k);
        t = classregtree(xtrainF,ytrainF,'minparent',size(xtrainF, 1)...
        ,'method','classification');
        l = eval(t,xtrain(indF == k, :));
        ll = zeros(size(l,1), 1);
        for i = 1:size(l,1)
            ll(i) = str2double(l{i});
        end
        res(indF == k) = ll;
    end
    err = sum((res ~= ytrain))/n;
end

