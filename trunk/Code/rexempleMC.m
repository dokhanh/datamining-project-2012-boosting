function [x,y] = rexempleMC(alpha, N, d, J)
%rexempleMC Generate multi-class dataset
%   Input:
%   alpha: Bayes Error
%   N: size of dataset
%   d: number of explaining variables
%   J: number of classes
%   Output:
%   x: explaining variables
%   y: response variable

    x = unifrnd(0, 1, N, d);
    y = zeros(N, 1);
    for i = 1:N
        if (x(i, 1)^2 + (x(i, 2) - 0.5)^2 < 2/(3*pi))
            y(i) = 1;
        else
            if ((x(i, 1) - 1)^2 + (x(i, 2) - 0.5)^2 < 2/(3*pi))
                y(i) = 2;
            else
                y(i) = 3;
            end
        end
    end
    %y = min(round(((x(:, 1) - 0.5).^2 + (x(:, 2) - 0.5).^2)*J*pi + 1), J);
    u = unifrnd(0, 1, N, 1);
    y(u < alpha) = unidrnd(J, sum(u < alpha), 1);
end

