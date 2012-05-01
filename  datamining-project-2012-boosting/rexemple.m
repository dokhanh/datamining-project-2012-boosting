function [x,y] = rexemple(alpha, N, d)
%rexemple sampling x et y, x variables explicatives, y variables de réponse
%   alpha erreur aléatoire

    x = unifrnd(0, 1, N, d);
    %y = 2*(2*x(:, 1) + x(:, 2) > 1.5)-1;
    y = 2*((x(:, 1) - 0.5).^2 + (x(:, 2) - 0.5).^2 > 1/6)-1;
    for i = 1:N
        u = unifrnd(0,1);
        if (u < alpha)
            y(i) = -y(i);
        end
    end
end

