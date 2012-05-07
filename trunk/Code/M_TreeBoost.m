function trees = M_TreeBoost(xtrain, ytrain, M)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    N = size(xtrain, 1);
    F = zeros(N, M);
    trees = cell(M, 3);
    
    alpha = 0.95;
    
    trees{1, 3} = median(ytrain);
    
    for m = 1:M
        if (m > 1)
            r_prev = ytrain - F(:, m-1);
        else
            %r_prev = ytrain - median(ytrain);
            r_prev = ytrain;
        end
        %calculate delta_m = quantile(r_prev)
        delta_m = quantile(r_prev, alpha);
        
        %ybarre
        ybarre = r_prev;
        ybarre(abs(r_prev) > delta_m) = delta_m*sign(r_prev(abs(r_prev) > delta_m));
        
        
        t = classregtree(xtrain,ybarre,'minparent',N);
        trees{m, 1} = t;
        [~, nodes] = eval(t, xtrain);
        r_barr = zeros(1,3);
        r_barr(2) = median(r_prev(nodes == 2));
        r_barr(3) = median(r_prev(nodes == 3));
        
        gamma = zeros(1,3);
        gamma(2) = r_barr(2) + 1/sum(nodes ==2)*sum(sign(r_prev(nodes == 2) - r_barr(2)).*min(delta_m, abs(r_prev(nodes == 2) - r_barr(2))));
        gamma(3) = r_barr(3) + 1/sum(nodes ==3)*sum(sign(r_prev(nodes == 3) - r_barr(3)).*min(delta_m, abs(r_prev(nodes == 3) - r_barr(3))));
        
        trees{m, 2} = gamma;
        
        h = zeros(N, 1);
        for i = 1:N
            h(i) = gamma(nodes(i));
        end
        
        if (m > 1)
            F(:, m) = F(:, m-1) + h;
        else
            %F(:, m) = h + median(ytrain);
            F(:, m) = h;
        end
    end
end