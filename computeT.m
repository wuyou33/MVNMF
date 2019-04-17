function [ T ] = computeT(X_tag)
%COMPUTET 此处显示有关此函数的摘要
%   此处显示详细说明
    m = size(X_tag, 1);
    T = zeros(m, m);
    
    for i=1:m
        frequency = sum(X_tag(i, :));
        if frequency ~= 0
            T(i, i)= 1 / frequency;
        else
            T(i, i) = 1e-10;
        end
    end
end

