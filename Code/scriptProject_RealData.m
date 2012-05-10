%%%%%%%%%%%%TEST ON REAL DATABASE%%%%%%%%%%%%%%
clear all;
Data = load('./RealDB/hill_valley_no_noise_training.data');

%remove the id
%Data = Data(:,2:11);
NData = size(Data, 1);
IdxClass = 101;

%5 fold cross-validation
K = 5;
M = 300; % nombre d'it?rations
errSynt1 = zeros(K, M);
errSynt2 = zeros(K, M);
errSynt3 = zeros(K, M);
errSynt4 = zeros(K, M);
errSynt5 = zeros(K, M);
errSynt6 = zeros(K, M);
for k = 1:K
    xtrain = Data(mod(1:NData, K) ~= (k-1),1:IdxClass-1);
    ytrain = Data(mod(1:NData, K) ~= (k-1),IdxClass);
    ytrain = ytrain - 3;

    xtest = Data(mod(1:NData, K) == (k-1),1:IdxClass-1);
    ytest = Data(mod(1:NData, K) == (k-1),IdxClass);
    ytest = ytest - 3;

   
    %[trees, err, C] = DiscreteAdaBoost(xtrain, ytrain, M);
    %trees = RealAdaBoost(xtrain, ytrain, M);
    trees4 = GentleAdaBoost(xtrain, ytrain, M);
    trees5 = LogitBoost_2class(xtrain, ytrain, M);
    % trees = AdaBoostMH(xtrain, ytrain, M, J);
    % trees = LogitBoost_Multiclass(xtrain, ytrain, M, J);
    trees1 = LS_Boost(xtrain, ytrain, M);
    trees3 = M_TreeBoost(xtrain, ytrain, M);
    trees2 = LAD_TreeBoost(xtrain, ytrain, M);
    trees6 = L2_TreeBoost(xtrain, ytrain, M);
    % trees1 = MultiClass_TreeBoost(xtrain, ytrain, M, J);

    n = size(xtest, 1);
    %[xtest, ytest] = rexempleMC(alpha, n, d, J);
    %resSynt = output_DiscreteAdaBoost(trees, C, M, xtest);
    %[resSynt, ~] = output_RealAdaBoost(trees, M, xtest);
    [resSynt4, FSynt4] = output_GentleAdaBoost(trees4, M, xtest);
    [resSynt5, FSynt5] = output_LogitBoost_2class(trees5, M, xtest);
%     [resSynt, ~] = output_AdaBoostMH(trees, M, xtest, J);
%     [resSynt, FSynt] = output_LogitBoost_Multiclass(trees, M, xtest, J);
    [resSynt1, FSynt1] = output_LS_Boost(trees1, M, xtest);
    [resSynt3, FSynt3] = output_M_TreeBoost(trees3, M, xtest);
    [resSynt2, FSynt2] = output_LAD_TreeBoost(trees2, M, xtest);
    [resSynt6, FSynt6] = output_L2_TreeBoost(trees6, M, xtest);
%     [resSynt1, FSynt1] = output_MultiClass_TreeBoost(trees1, M, xtest, J);

    for m = 1:M
        errSynt1(k, m) = sum(resSynt1(:, m) ~= ytest)/n;
        errSynt2(k, m) = sum(resSynt2(:, m) ~= ytest)/n;
        errSynt3(k, m) = sum(resSynt3(:, m) ~= ytest)/n;
        errSynt4(k, m) = sum(resSynt4(:, m) ~= ytest)/n;
        errSynt5(k, m) = sum(resSynt5(:, m) ~= ytest)/n;
        errSynt6(k, m) = sum(resSynt6(:, m) ~= ytest)/n;
    end
end
errSyntt1 = mean(errSynt1);
errSyntt2 = mean(errSynt2);
errSyntt3 = mean(errSynt3);
errSyntt4 = mean(errSynt4);
errSyntt5 = mean(errSynt5);
errSyntt6 = mean(errSynt6);
%%
close all;
figure;
plot(errSyntt1(:, 1:250), 'cyan'); hold on
plot(errSyntt2(:, 1:250), 'magenta'); hold on
plot(errSyntt3(:, 1:250), 'black'); hold on
plot(errSyntt4(:, 1:250), 'red'); hold on
plot(errSyntt5(:, 1:250), 'green'); hold on
plot(errSyntt6(:, 1:250), 'blue'); hold on
hold off
xlabel('Iteration', 'FontSize', 15);
ylabel('Error Rate', 'FontSize', 15);
set(gca,'fontsize',15);