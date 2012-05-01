function trees = RealAdaBoost(xtrain, ytrain, M)
%RealAdaBoost Exécuter l'algo Real AdaBoost, retourner un ensemble de
%classifieurs de base avec des coefficients correspondants
%   Input:
%   xtrain, ytrain: data set pour entraînement
%   M: le nombre d'itérations
%   Output:
%   trees: un cell contenant des classifieurs de base
%   err: erreurs
%   C: des coefficients correspondants aux classifieurs obtenus

    N = size(xtrain, 1);
    w = (1/N)*ones(N,1);
    trees = cell(M,1);
    err = zeros(M,1); % error at each iteration
    C = zeros(M,1);
    
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
        f = (1/2)*log(p./(1-p));
        w = w.*exp(-ytrain.*f);
        w = w/sum(w);
    end
end

