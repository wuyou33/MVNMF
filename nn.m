function [ indices ] = nn( N, dis )
%KNN �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    viewNum = size(dis ,1);
    dis = sum(dis, 1)/ viewNum;
    [~, indices] = sort(dis);
    indices = indices(1:N);
end

