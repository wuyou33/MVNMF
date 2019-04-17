function [ T ] = computeT(X_tag)
%COMPUTET �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
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

