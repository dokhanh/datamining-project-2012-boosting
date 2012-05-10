%%%%%%%%%%%%TEST ON REAL DATABASE%%%%%%%%%%%%%%
clear all;
Data = load('./RealDB/winequality-red.csv');

%remove the id
%Data = Data(:,2:11);
NData = size(Data, 1);
IdxClass = 12;
Data(:,IdxClass) = Data(:,IdxClass) - min(Data(:,IdxClass)) + 1;
J = max(Data(:,IdxClass)); %number of class;

%5 fold cross-validation
K = 5;
M = 100; % nombre d'it?rations
errSynt1 = zeros(K, M);
errSynt2 = zeros(K, M);
errSynt3 = zeros(K, M);
for k = 1:K
    xtrain = Data(mod(1:NData, K) ~= (k-1),1:IdxClass-1);
    ytrain = Data(mod(1:NData, K) ~= (k-1),IdxClass);
    %ytrain = ytrain - 3;

    xtest = Data(mod(1:NData, K) == (k-1),1:IdxClass-1);
    ytest = Data(mod(1:NData, K) == (k-1),IdxClass);
    %ytest = ytest - 3;

    trees1 = AdaBoostMH(xtrain, ytrain, M, J, 1);
    trees2 = LogitBoost_Multiclass(xtrain, ytrain, M, J);
    trees3 = MultiClass_TreeBoost(xtrain, ytrain, M, J);


    n = size(xtest, 1);
    [resSynt1, ~] = output_AdaBoostMH(trees1, M, xtest, J, 1);
    [resSynt2, ~] = output_LogitBoost_Multiclass(trees2, M, xtest, J);
    [resSynt3, ~] = output_MultiClass_TreeBoost(trees3, M, xtest, J);

    for m = 1:M
        errSynt1(k, m) = sum(resSynt1(:, m) ~= ytest)/n;
        errSynt2(k, m) = sum(resSynt2(:, m) ~= ytest)/n;
        errSynt3(k, m) = sum(resSynt3(:, m) ~= ytest)/n;
    end
end
errSyntt1 = mean(errSynt1);
errSyntt2 = mean(errSynt2);
errSyntt3 = mean(errSynt3);
%%
close all;
figure;
plot(errSyntt1, 'red'); hold on
plot(errSyntt2, 'green'); hold on
plot(errSyntt3, 'blue'); hold on
hold off
xlabel('Iteration', 'FontSize', 15);
ylabel('Error Rate', 'FontSize', 15);
set(gca,'fontsize',15);