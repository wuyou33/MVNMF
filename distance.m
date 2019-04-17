function [ d ] = distance(q, X, j)
%计算某一视图下，查询图像的特征q与数据集中所有图片的距离
%   此处显示详细说明
	error = bsxfun(@minus, X, q);
    summ = max(bsxfun(@plus, X, q), 1e-10);
	if (1<=j) && (j<=8) 
        d = sum((error.^2)./ (2*summ), 1);                %sum( (xi-yi)^2 / (xi+yi) ) / 2;
    else
        if j==9
        	d = sqrt(sum(error.^2, 1));
        else 
            d = sum(abs(error), 1);
        end
	end
	d = d ./ sum(d);
end

