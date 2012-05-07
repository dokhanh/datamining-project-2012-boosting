function trees = ModestAdaBoost(xtrain, ytrain, M)
%
%
%
    N = size(xtrain, 1);
    
    %w = (1/N)*ones(N,1);
    w = ytrain;
    w(w==1) = 0.5/sum(w==1);
    w(w==-1) = 0.5/sum(w==-1);
    
    trees = cell(M,1);
    err = zeros(M,1); % error at each iteration
    C = zeros(M,1);
    
    for m = 1:M
        m
        %for current distribution
        t = classregtree(xtrain,ytrain,'minparent',N,'weights',w,'method','classification');
        trees{m} = t;
        [ytrained, nodesTrained] = eval(t, xtrain);
        
        
        %update fm
        f = (p.*(1-p_inv) - (1-p).*p_inv);%*1/2;
        
        %update distribution
        w = w.*exp(-ytrain.*f);
        w = w/sum(w);
    end
end