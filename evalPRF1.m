function [precision, recall, f1] = evalPRF1( Ytest, Tag_predict )
%SCORE 此处显示有关此函数的摘要
%   此处显示详细说明
    [nClass, nSmp] = size(Ytest);
    precision = 0;
    recall = 0;
    for i  = 1:nSmp
        Y = Ytest(:, i);
        predict = Tag_predict(:, i);
        tp = sum(Y(find(predict==1)));
        fp = numel(Y(find(predict==1))) - tp;

        fn = sum(Y(find(predict==0)));
        tn = numel(Y(find(predict==0))) - fn;

        p = tp / (tp+fp);
        r = tp / (tp+fn);
        precision = precision + p;
        recall = recall + r;
    end

    precision = precision / nSmp;
    recall = recall / nSmp;
    f1 = 2 * precision * recall / (precision + recall);
end

