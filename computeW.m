function [ W ] = computeW( X_tag)
%COMPUTEW �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    [~, n] = size(X_tag);
    W = zeros(n, 1);
    for i=1:n
        fre = sum(sum(X_tag(find(X_tag(:, i)==1), :)));
        if fre == 0
            W(i) = 1e-10;
        else
            W(i) = 1 / fre;
        end
    end
    W = diag(W);
end

