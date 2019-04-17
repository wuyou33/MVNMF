function [U_final, V_final, nIter_final, elapse_final, bSuccess, objhistory_final] = PerViewNMF(X, k, Vo, options, U, V, W, T, view)
% This is a module of Multi-View Non-negative Matrix Factorization
% (MultiNMF) for the update for one view as in lines 5-9 in Alg. 1
%
% Notation:
% X ... (mFea x nSmp) data matrix of one view
%       mFea  ... number of features
%       nSmp  ... number of samples
% k ... number of hidden factors
% Vo... consunsus
% options ... Structure holding all settings
% U ... initialization for basis matrix 
% V ... initialization for coefficient matrix 
%
%   Originally written by Deng Cai (dengcai AT gmail.com) for GNMF
%   Modified by Jialu Liu (jliu64@illinois.edu)

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIterOrig = options.minIter;
minIter = minIterOrig-1;
meanFitRatio = options.meanFitRatio;
alpha = options.alpha(view);

% nIter_final = 0;
% U_final = U;
% V_final = V;
% objhistory = 0; 
% objhistory_final = 0;
% elapse_final = 0;

Norm = 1;
NormV = 0;

[mFea,nSmp]=size(X);

bSuccess.bSuccess = 1;
selectInit = 1;
if isempty(U)
    U = abs(rand(mFea,k));
    V = abs(rand(nSmp,k));
else
    nRepeat = 1;
end

[U,V] = Normalize(U, V);
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, Vo, alpha, W, T);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, Vo, alpha, W, T);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end
    
W2 = W*W';
T2 = T'*T;
    
tryNo = 0;
oldobj = 0;
newobj = 0;
while tryNo < nRepeat   
    tmp_T = cputime;
    tryNo = tryNo+1;
    nIter = 0;
    maxErr = 1;
    nStepTrial = 0;
    %disp a
    while(maxErr > differror)
        oldobj = newobj;
        % ===================== update V ========================
        temp1 = W2'*X'*T2*U + alpha*W2*Vo;  
        if view == 16
            temp2 = W2*V*U'*T2*U + alpha*W2*V + ones(nSmp, mFea)*U;  
        else
            temp2 = W2*V*U'*T2*U + alpha*W2*V;
        end
        
        V = V.*(temp1 ./ max(temp2,1e-10));
        % ===================== update U ========================
        temp1 = bsxfun(@plus, T2*X*W2*V, alpha*sum(diag(W2)'*(V.*Vo), 1));
        if view == 16
            temp2 = bsxfun(@plus, T2*U*V'*W2*V, alpha*sum(U, 1).*sum((diag(W2)'*(V.^2)), 1)) + ones(mFea, nSmp)*V;
        else
            temp2 = bsxfun(@plus, T2*U*V'*W2*V, alpha*sum(U, 1).*sum((diag(W2)'*(V.^2)), 1));
        end
        %temp1 = T2*X*W2*V;
        %temp2 = T2*U*V'*W2*V; 
        U = U.*(temp1 ./ max(temp2,1e-10)); 
        

        
        [U,V] = Normalize(U, V);
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, Vo, alpha, W, T);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, Vo, alpha, W, T);
                    objhistory = [objhistory newobj]; 
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;  
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V, Vo, alpha ,W , T);
                        objhistory = [objhistory newobj]; 
                    end
                    newobj = CalculateObj(X, U, V, Vo, alpha ,W , T);
                    maxErr = abs(newobj - oldobj);
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
                        else
                            objhistory = 0;
                        end
                    end
                end
            end
        end
    end
    
    elapse = cputime - tmp_T;
    objhistory = 0;
    
    if tryNo == 1
        U_final = U;
        V_final = V;
        nIter_final = nIter;
        elapse_final = elapse;
        objhistory_final = objhistory;
        bSuccess.nStepTrial = nStepTrial;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           V_final = V;
           nIter_final = nIter;
           objhistory_final = objhistory;
           bSuccess.nStepTrial = nStepTrial;
           if selectInit
               elapse_final = elapse;
           else
               elapse_final = elapse_final+elapse;
           end
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(mFea,k));
            V = abs(rand(nSmp,k));
            [U,V] = Normalize(U, V);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

nIter_final = nIter_final + minIterOrig;
[U_final, V_final] = Normalize(U_final, V_final);



%==========================================================================

function [obj, dV] = CalculateObj(X, U, V, L, alpha, W, T, deltaVU, dVordU)
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    dV = [];
    maxM = 62500000;
    [~, nSmp] = size(X);
    mn = numel(X);
    nBlock = floor(mn*3/maxM);

    if mn < maxM
        dX = T*(U*V'-X)*W;
        obj_NMF = sum(sum(dX.^2));
        if deltaVU
            if dVordU
                dV = dX'*U + L*V;
            else
                dV = dX*V;
            end
        end
    else
        obj_NMF = 0;
        if deltaVU
            if dVordU
                dV = zeros(size(V));
            else
                dV = zeros(size(U));
            end
        end
        for i = 1:ceil(nSmp/nBlock)
            if i == ceil(nSmp/nBlock)
                smpIdx = (i-1)*nBlock+1:nSmp;
            else
                smpIdx = (i-1)*nBlock+1:i*nBlock;
            end
            dX = U*V(smpIdx,:)'-X(:,smpIdx);
            obj_NMF = obj_NMF + sum(sum(dX.^2));
            if deltaVU
                if dVordU
                    dV(smpIdx,:) = dX'*U;
                else
                    dV = dU+dX*V(smpIdx,:);
                end
            end
        end
        if deltaVU
            if dVordU
                dV = dV + L*V;
            end
        end
    end
    tmp = W'*(V-L);
    obj_Lap = sum(sum(tmp.^2));
   
    dX = T*(U*V'-X)*W;
    obj_NMF = sum(sum(dX.^2));
    obj = obj_NMF+ alpha * obj_Lap;


function [U, V] = Normalize(U, V)
    [U,V] = NormalizeUV(U, V, 0, 1);

function [U, V] = NormalizeUV(U, V, NormV, Norm)
    nSmp = size(V,1);
    mFea = size(U,1);
    if Norm == 2
        if NormV
            norms = sqrt(sum(V.^2,1));
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sqrt(sum(U.^2,1));
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = V.*repmat(norms,nSmp,1);
        end
    else
        if NormV
            norms = sum(abs(V),1);
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sum(abs(U),1);
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = bsxfun(@times, V, norms);
        end
    end

        