%% créer le data set
alpha = 0; % erreurs aléatoires
N = 500;
d = 2; % nombre de variables explicatives
[xtrain,ytrain] = rexemple(alpha, N, d);

%% Discrete AdaBoost
M = 100; % nombre d'itérations
%[trees, err, C] = DiscreteAdaBoost(xtrain, ytrain, M);
trees = RealAdaBoost(xtrain, ytrain, M);

%% Output
e = 100; % nombre d'essais
n = 100;
errSynt = zeros(e, M);
for et = 1:e
    et
    [xtest, ytest] = rexemple(alpha, n, d);
    %resSynt = output_DiscreteAdaBoost(trees, C, M, xtest);
    [resSynt, ~] = output_RealAdaBoost(trees, M, xtest);
    for m = 1:M
        errSynt(et, m) = sum(resSynt(:, m) ~= ytest)/n;
    end
end
errSyntt = mean(errSynt);
plot(errSyntt);