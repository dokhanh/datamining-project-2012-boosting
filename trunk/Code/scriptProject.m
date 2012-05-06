%% cr?er le data set
alpha = 0; % erreurs al?atoires
N = 500;
d = 2; % nombre de variables explicatives
%[xtrain,ytrain] = rexemple(alpha, N, d);
J = 4;
[xtrain,ytrain] = rexempleMC(alpha, N, d, J);

%% Discrete AdaBoost
M = 100; % nombre d'it?rations
%[trees, err, C] = DiscreteAdaBoost(xtrain, ytrain, M);
%trees = RealAdaBoost(xtrain, ytrain, M);
% trees2 = GentleAdaBoost(xtrain, ytrain, M);
% trees = LogitBoost_2class(xtrain, ytrain, M);
trees = AdaBoostMH(xtrain, ytrain, M, J);

%% Output
e = 100; % nombre d'essais
n = 100;
errSynt = zeros(e, M);
errSynt2 = zeros(e, M);
for et = 1:e
    et
    [xtest, ytest] = rexempleMC(alpha, n, d, J);
    %resSynt = output_DiscreteAdaBoost(trees, C, M, xtest);
    %[resSynt, ~] = output_RealAdaBoost(trees, M, xtest);
%     [resSynt2, ~] = output_GentleAdaBoost(trees2, M, xtest);
%     [resSynt, ~] = output_LogitBoost_2class(trees, M, xtest);
    [resSynt, F, R] = output_AdaBoostMH(trees, M, xtest, J);
    for m = 1:M
        errSynt(et, m) = sum(resSynt(:, m) ~= ytest)/n;
    end
end
errSyntt = mean(errSynt);
%errSyntt2 = mean(errSynt2);
plot(errSyntt, 'b');
%hold on
%plot(errSyntt2);
