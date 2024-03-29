%% cr?er le data set
alpha = 0.1; % erreurs al?atoires
N = 500;
d = 2; % nombre de variables explicatives
% [xtrain,ytrain] = rexemple(alpha, N, d);
J = 4;
[xtrain,ytrain] = rexempleMC(alpha, N, d, J);

%% Discrete AdaBoost
M = 100; % nombre d'it?rations
% [trees1, err, C] = DiscreteAdaBoost(xtrain, ytrain, M);
% trees2 = RealAdaBoost(xtrain, ytrain, M);
% trees3 = GentleAdaBoost(xtrain, ytrain, M);
% trees = LogitBoost_2class(xtrain, ytrain, M);
% trees1 = AdaBoostMH(xtrain, ytrain, M, J, 1);
% trees2 = AdaBoostMH(xtrain, ytrain, M, J, 2);
% trees3 = AdaBoostMH(xtrain, ytrain, M, J, 3);
% trees4 = AdaBoostMH(xtrain, ytrain, M, J, 4);
% trees5 = AdaBoostMH(xtrain, ytrain, M, J, 5);
% trees6 = AdaBoostMH(xtrain, ytrain, M, J, 6);
% trees = LogitBoost_Multiclass(xtrain, ytrain, M, J);
% trees = LS_Boost(xtrain, ytrain, M);
% trees = LAD_TreeBoost(xtrain, ytrain, M);
% trees = L2_TreeBoost(xtrain, ytrain, M);
% trees1 = MultiClass_TreeBoost(xtrain, ytrain, M, J);
trees1 = AdaBoostMH(xtrain, ytrain, M, J, 1);
trees2 = LogitBoost_Multiclass(xtrain, ytrain, M, J);
trees3 = MultiClass_TreeBoost(xtrain, ytrain, M, J);

%% Output
e = 100; % nombre d'essais
n = 100;
errSynt1 = zeros(e, M);
errSynt2 = zeros(e, M);
errSynt3 = zeros(e, M);
errSynt4 = zeros(e, M);
errSynt5 = zeros(e, M);
errSynt6 = zeros(e, M);
for et = 1:e
    et
%     [xtest, ytest] = rexemple(alpha, n, d);
    [xtest, ytest] = rexempleMC(alpha, n, d, J);
%     resSynt1 = output_DiscreteAdaBoost(trees1, C, M, xtest);
%     [resSynt2, ~] = output_RealAdaBoost(trees2, M, xtest);
%     [resSynt3, ~] = output_GentleAdaBoost(trees3, M, xtest);
%     [resSynt, ~] = output_LogitBoost_2class(trees, M, xtest);
%     [resSynt1, ~] = output_AdaBoostMH(trees1, M, xtest, J, 1);
%     [resSynt2, ~] = output_AdaBoostMH(trees2, M, xtest, J, 2);
%     [resSynt3, ~] = output_AdaBoostMH(trees3, M, xtest, J, 3);
%     [resSynt4, ~] = output_AdaBoostMH(trees4, M, xtest, J, 4);
%     [resSynt5, ~] = output_AdaBoostMH(trees5, M, xtest, J, 5);
%     [resSynt6, ~] = output_AdaBoostMH(trees6, M, xtest, J, 6);
%     [resSynt, FSynt] = output_LogitBoost_Multiclass(trees, M, xtest, J);
%     [resSynt, FSynt] = output_LS_Boost(trees, M, xtest);
%     [resSynt, FSynt] = output_LAD_TreeBoost(trees, M, xtest);
%     [resSynt, FSynt] = output_L2_TreeBoost(trees, M, xtest);
%     [resSynt1, FSynt1] = output_MultiClass_TreeBoost(trees1, M, xtest, J);
    [resSynt1, ~] = output_AdaBoostMH(trees1, M, xtest, J, 1);
    [resSynt2, ~] = output_LogitBoost_Multiclass(trees2, M, xtest, J);
    [resSynt3, ~] = output_MultiClass_TreeBoost(trees3, M, xtest, J);
    for m = 1:M
        errSynt1(et, m) = sum(resSynt1(:, m) ~= ytest)/n;
        errSynt2(et, m) = sum(resSynt2(:, m) ~= ytest)/n;
        errSynt3(et, m) = sum(resSynt3(:, m) ~= ytest)/n;
%         errSynt4(et, m) = sum(resSynt4(:, m) ~= ytest)/n;
%         errSynt5(et, m) = sum(resSynt5(:, m) ~= ytest)/n;
%         errSynt6(et, m) = sum(resSynt6(:, m) ~= ytest)/n;
%         errSynt2(et, m) = sum(resSynt2(:, m) ~= ytest)/n;
%         errSynt3(et, m) = sum(resSynt3(:, m) ~= ytest)/n;
    end
end
errSyntt1 = mean(errSynt1);
errSyntt2 = mean(errSynt2);
errSyntt3 = mean(errSynt3);
% errSyntt4 = mean(errSynt4);
% errSyntt5 = mean(errSynt5);
% errSyntt6 = mean(errSynt6);
% errSyntt2 = mean(errSynt2);
% errSyntt3 = mean(errSynt3);
plot(errSyntt1, 'r');
hold on
plot(errSyntt2, 'c');
plot(errSyntt3, 'm');
% plot(errSyntt4, 'k');
% plot(errSyntt5, 'g');
% plot(errSyntt6, 'b');
% hold on
% plot(errSyntt2, 'r');
% hold on
% plot(errSyntt3, 'k');
