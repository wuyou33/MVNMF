function [ d ] = distance(q, X, j)
%����ĳһ��ͼ�£���ѯͼ�������q�����ݼ�������ͼƬ�ľ���
%   �˴���ʾ��ϸ˵��
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

