function [ indices ] = nn( N, dis )
%KNN 此处显示有关此函数的摘要
%   此处显示详细说明
    viewNum = size(dis ,1);
    dis = sum(dis, 1)/ viewNum;
    [~, indices] = sort(dis);
    indices = indices(1:N);
end

