clear 

addpath(genpath(pwd));

N = 40;K =50;
viewNum = 16;
method = 'origin';
dataset = 'corel5k';

options = [];
options.maxIter = 200;
options.error = 1e-4;
options.nRepeat = 30;
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 20;
options.alpha = [];

%load data
switch dataset
    case 'corel5k'
        D_1 = double(vec_read('corel5k\corel5k_train_DenseHue.hvecs')');
        D_2 = double(vec_read('corel5k\corel5k_train_DenseHueV3H1.hvecs')');
        D_3 = double(vec_read('corel5k\corel5k_train_HarrisHue.hvecs')');
        D_4 = double(vec_read('corel5k\corel5k_train_HarrisHueV3H1.hvecs')');
        D_5 = double(vec_read('corel5k\corel5k_train_DenseSift.hvecs')');
        D_6 = double(vec_read('corel5k\corel5k_train_DenseSiftV3H1.hvecs')');
        D_7 = double(vec_read('corel5k\corel5k_train_HarrisSift.hvecs')');
        D_8 = double(vec_read('corel5k\corel5k_train_HarrisSiftV3H1.hvecs')');
        D_9 = double(vec_read('corel5k\corel5k_train_Gist.fvec')');
        D_10 = double(vec_read('corel5k\corel5k_train_Rgb.hvecs32')');
        D_11 = double(vec_read('corel5k\corel5k_train_RgbV3H1.hvecs32')');
        D_12 = double(vec_read('corel5k\corel5k_train_Lab.hvecs32')');
        D_13 = double(vec_read('corel5k\corel5k_train_LabV3H1.hvecs32')');
        D_14 = double(vec_read('corel5k\corel5k_train_Hsv.hvecs32')');
        D_15 = double(vec_read('corel5k\corel5k_train_HsvV3H1.hvecs32')');
        Ytrain = double(vec_read('corel5k\corel5k_train_annot.hvecs')');
        
        Q_1 = double(vec_read('corel5k\corel5k_test_DenseHue.hvecs')');
        Q_2 = double(vec_read('corel5k\corel5k_test_DenseHueV3H1.hvecs')');
        Q_3 = double(vec_read('corel5k\corel5k_test_HarrisHue.hvecs')');
        Q_4 = double(vec_read('corel5k\corel5k_test_HarrisHueV3H1.hvecs')');
        Q_5 = double(vec_read('corel5k\corel5k_test_DenseSift.hvecs')');
        Q_6 = double(vec_read('corel5k\corel5k_test_DenseSiftV3H1.hvecs')');
        Q_7 = double(vec_read('corel5k\corel5k_test_HarrisSift.hvecs')');
        Q_8 = double(vec_read('corel5k\corel5k_test_HarrisSiftV3H1.hvecs')');
        Q_9 = double(vec_read('corel5k\corel5k_test_Gist.fvec')');
        Q_10 = double(vec_read('corel5k\corel5k_test_Rgb.hvecs32')');
        Q_11 = double(vec_read('corel5k\corel5k_test_RgbV3H1.hvecs32')');
        Q_12 = double(vec_read('corel5k\corel5k_test_Lab.hvecs32')');
        Q_13 = double(vec_read('corel5k\corel5k_test_LabV3H1.hvecs32')');
        Q_14 = double(vec_read('corel5k\corel5k_test_Hsv.hvecs32')');
        Q_15 = double(vec_read('corel5k\corel5k_test_HsvV3H1.hvecs32')');
        Ytest = double(vec_read('corel5k\corel5k_test_annot.hvecs')');
    case 'espgame'
        D_1 = double(vec_read('espgame\espgame_train_Gist.fvec'));
        D_2 = double(vec_read('espgame\espgame_train_DenseHue.hvecs'));
        D_3 = double(vec_read('espgame\espgame_train_DenseHueV3H1.hvecs'));
        D_4 = double(vec_read('espgame\espgame_train_DenseSift.hvecs'));
        D_5 = double(vec_read('espgame\espgame_train_DenseSiftV3H1.hvecs'));
        D_6 = double(vec_read('espgame\espgame_train_HarrisHue.hvecs'));
        D_7 = double(vec_read('espgame\espgame_train_HarrisHueV3H1.hvecs'));
        D_8 = double(vec_read('espgame\espgame_train_Hsv.hvecs32'));
        D_9 = double(vec_read('espgame\espgame_train_HsvV3H1.hvecs32'));
        D_10 = double(vec_read('espgame\espgame_train_Lab.hvecs32'));
        D_11 = double(vec_read('espgame\espgame_train_LabV3H1.hvecs32'));
        D_12 = double(vec_read('espgame\espgame_train_Rgb.hvecs32'));
        D_13 = double(vec_read('espgame\espgame_train_RgbV3H1.hvecs32'));
        D_14 = double(vec_read('espgame\espgame_train_HarrisSift.hvecs'));
        D_15 = double(vec_read('espgame\espgame_train_HarrisSiftV3H1.hvecs'));     
end

test_n = size(Q_1, 2);%size(Q_1, 2);

for i=1:viewNum-1
    options.alpha = [options.alpha; 0.01];
end
options.alpha = [options.alpha; 1];
train_D = cell(1, viewNum);
test_Q = cell(1, viewNum);
X = cell(1, viewNum);
q = cell(1, viewNum-1);
for i=1:viewNum-1
     train_D{i} = eval(['D_',num2str(i)]);
     test_Q{i} = eval(['Q_',num2str(i)]);
end
train_D{viewNum} = Ytrain;
test_Q{viewNum} = Ytest;

train_n = size(D_1, 2);

dis = zeros(viewNum-1, train_n);
final_U = [];
final_V = [];
cen_V = [];
tag_num = size(Ytrain, 1);
tag_predict = zeros(tag_num, test_n);
x_tag = zeros(tag_num, test_n);
thre = [];
for i = 1:test_n
    disp([num2str(i), '-th sample']);
    tic
    
    for j=1:viewNum-1
       q{j} = test_Q{j}(:, i); 
       dis(j, :) = distance(q{j}, train_D{j}, j);
    end
    knn_indices = nn(N, dis);
    for j=1:viewNum-1
       X{j} = train_D{j}(:, knn_indices);
    end
    X{viewNum} = Ytrain(:, knn_indices);
%     for j=1:viewNum
%        X{j} = mapminmax(X{j}, 0, 1);
%     end
    W = computeW(X{viewNum});

        [final_U, final_V, cen_V, T] = mvnmf(X, K, options, W, method); 
        
%        avg_V = 0;
%         for k=1:viewNum-1
%             avg_V = avg_V + options.alpha(k) * final_V{k};
%         end
%         X_recover = final_U{viewNum} * avg_V';
%         thre(i) = min(X_recover(find(X{viewNum}==1)));
        x_tag(:, i) = predict1(final_U, final_V, cen_V, q);
        [~, tag_indices] = sort(x_tag(:, i), 'descend');
        tag_predict(tag_indices(1:5), i) = 1;
    toc

%      else
%         [final_U, final_V, cen_V] = mvnmf_m(X, K, options, W); 
%     end
end
%threshold = mean(thre);
%tag_predict(find(x_tag>=threshold)) = 1;
[ precision, recall, f1 ] = evalPRF1( Ytest, tag_predict);
precision
recall
f1







