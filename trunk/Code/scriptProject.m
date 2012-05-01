%% cr?er le data set
alpha = 0; % erreurs al?atoires
N = 500;
d = 2; % nombre de variables explicatives
[xtrain,ytrain] = rexemple(alpha, N, d);

%% Discrete AdaBoost
M = 100; % nombre d'it?rations
%[trees, err, C] = DiscreteAdaBoost(xtrain, ytrain, M);
trees = RealAdaBoost(xtrain, ytrain, M);
trees2 = GentleAdaBoost(xtrain, ytrain, M);

%% Output
e = 100; % nombre d'essais
n = 100;
errSynt = zeros(e, M);
errSynt2 = zeros(e, M);
for et = 1:e
    et
    [xtest, ytest] = rexemple(alpha, n, d);
    %resSynt = output_DiscreteAdaBoost(trees, C, M, xtest);
    [resSynt, ~] = output_RealAdaBoost(trees, M, xtest);
    [resSynt2, ~] = output_GentleAdaBoost(trees2, M, xtest);
    for m = 1:M
        errSynt(et, m) = sum(resSynt(:, m) ~= ytest)/n;
        errSynt2(et, m) = sum(resSynt2(:, m) ~= ytest)/n;
    end
end
errSyntt = mean(errSynt);
errSyntt2 = mean(errSynt2);
plot(errSyntt);
figure;
plot(errSyntt2);
