function [ x_tagpredict ] = predict1( U, V, Vc, q)
%PREDICT1 此处显示有关此函数的摘要
%   此处显示详细说明
    
    viewNum = length(U);
    d = zeros(viewNum - 1, 1);
    for i=1:viewNum - 1
        error = V{i} - Vc;
        d = sum(sum(error.^2));
    end
    lambda = d / sum(d);
    for j=1:viewNum - 1
        [vq(:, j), ~, ~, ~, ~, ~] = GPSR_BB(q{j}, U{j}, 0.1*max(max(U{j}'*q{j})));
    end
    v_tag = vq * lambda;
    x_tagpredict = U{viewNum} * v_tag;
end

