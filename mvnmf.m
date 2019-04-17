function [ U, V, centroidV, T ] = mvnmf(X, K, options, W, method)
%MVNMF 此处显示有关此函数的摘要
%   此处显示详细说明
    
    viewNum = length(X);
    Rounds = options.rounds;
    tol = 3;
    
    lamda = options.alpha;
    U = cell(1, viewNum);
    V = cell(1, viewNum);
    U_ = {};
    V_ = {};
    log = zeros(Rounds, 1);
    oldL = 1000;
    den = 0;
    
    j = 0;
    while j<3
        j = j + 1;
        if j == 1
            [U{1}, V{1}] = NMF(X{1}, K, options, U_, V_);
        else
            [U{1}, V{1}] = NMF(X{1}, K, options, U_, V{viewNum});
        end
        for i = 2:viewNum
            [U{i}, V{i}] = NMF(X{i}, K, options, U_, V{i-1});
        end
    end
    

    for i=1:viewNum-1
        den = den + lamda(i) * W*W';
    end
    den = diag(1 ./ max(diag(den), 1e-10));
    
    j = 0;
    count = 0;
    while j<Rounds
        j = j + 1;
        disp([num2str(j),'-th round']);
        
        centroidV = 0;
        for i = 1:viewNum-1
            centroidV = centroidV + lamda(i) * W*W'*V{i};
        end
        centroidV = den * centroidV;

        
        logL = 0;
        for i = 1:viewNum
            if i ~= viewNum
                mFea = size(X{i}, 1);
                T = eye(mFea);
            else
                T = computeT(X{i}); 
            end
            tmp1 = T*(X{i} - U{i}*V{i}')*W;
            tmp2 = W*(V{i} - centroidV);
            tmp1 = sum(sum(tmp1.^2));
            tmp2 = lamda(i) * sum(sum(tmp2.^2));
            if i == viewNum
                tmp3 = sum(sum(U{i}*V{i}'));
            else
                tmp3 = 0;
            end
            logL = logL + tmp1 + tmp2 + tmp3;
        end
        log(end+1) = logL;
        
        disp(logL);
        if j ~= 1
            if(oldL < logL)
                count = count + 1;
                if count == tol
                    return;
                end
                U = oldU;
                V = oldV;
                logL = oldL;
                disp('restrart this iteration');
            else
                if abs(oldL-logL)<=options.error
                    return;
                end 
            end
        end
        oldU = U;
        oldV = V;
        oldL = logL;
    
        for i = 1:viewNum
            if i ~= viewNum
                mFea = size(X{i}, 1);
                T = eye(mFea);
            else
                T = computeT(X{i}); 
            end
            disp(['updating of ', num2str(i), '-th view']);
            if method == 'origin'
                [U{i}, V{i}] = PerViewNMF(X{i}, K, centroidV, options, U{i}, V{i}, W, T, i);
            else
                
            end
        end
    end
end

