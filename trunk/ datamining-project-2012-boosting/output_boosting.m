function errSyntt = output_boosting(trees, C, e, alpha, n, d, M)
%output_boosting Return outputs of boosting methods on artificial data set
%   Input:
%   trees: the set of base classifiers
%   C: coefficients of each base classifier
%   e: number of tries
%   n, d: size of test dataset
%   M: number of iterations
%   Output:
%   errSyntt: estimated errors after each iteration

    errSynt = zeros(e, M);
    for et = 1:e
        et
        [xtest, ytest] = rexemple(alpha, n, d);
        resSynt = output_DiscreteAdaBoost(trees, C, M, xtest);
        for m = 1:M
            errSynt(et, m) = sum(resSynt(:, m) ~= ytest)/n;
        end
    end
    errSyntt = mean(errSynt);
end

