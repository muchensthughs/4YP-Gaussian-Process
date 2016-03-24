%%%%%%%%%% Initialisation %%%%%%%%%%%%%%%
clear all



addpath(genpath('/Users/muchen/Desktop/Gaussian-Process/code/occ1990dd'));


filename1 = '/Users/muchen/Desktop/DA_training_data/1990_all.xlsx';
A = xlsread(filename1);


filename2 = '/Users/muchen/Desktop/Gaussian-Process/code/1990results.csv';



filename4 = '/Users/muchen/Desktop/Gaussian-Process/code/data_matching.xlsx';


X_all = A(:,2:end-1);
%ind = [1,2,3];
ind = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26];
%ind = [1,2,3,4,15];
X_all = feature_select(X_all,ind);

 num_dims = size(X_all,2);
 mean_A = mean (X_all);
 num_inst = length(X_all(:,1));
 
 sig_func = 'logistic';
 %%%%%%%%%%%%%%%%%%%preprocessing%%%%%%%%%%%%%%%%%%
 for j = 1:num_dims,
     X_all(:,j) = X_all(:,j) - mean_A(1,j);     
     var = sum(X_all(:,j).^2)/num_inst;
     X_all(:,j) = X_all(:,j)./sqrt(var);
 end
 
 %%%%%%%%%%%%%PCA%%%%%%%%%%%%%%%%%%%%
 [coeff, score, latent] = pca(X_all);
  varorder = cumsum(latent)./sum(latent);
%   %preserve 95% variance
%  %coeffinuse = coeff(:,1:19);
%  %X_all = X_all * coeffinuse;
  X_all = score(:,1:19);
 num_dims = size(X_all,2);
 numSamples = 200;
 
 X = [];
Y_01 = [];
for i = 1:length(A(:,1)),
    if ~isnan(A(i,end)) 
        X = [X; X_all(i,:)];
        Y_01 = [Y_01; A(i,end)];
    end
end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class = []; 
Y = Y_01;

for i = 1:length(Y_01)
    if Y_01(i) == 0,
        Y(i) = -1;
    end
end

train_length = size(X,1);

AUC_all = [];
training_AUC_all = [];
optimised_params_all = [];
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%For A%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


weights = zeros(1,num_dims);
change_weights = ones(1,num_dims);
params = [-1, 0.3, 0, weights];
change_vars = [1, 1, 0, change_weights];



 

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%For A%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_test = 8;
for k = 1:3,    
r = randsample(train_length,num_test);
 [rest] = setdiff((1:train_length),r,'stable');
 train_X = X(rest,:);
 train_Y = Y(rest,:);
 test_X = X(r,:);
 test_Y = Y_01(r,:);
 test_Y = test_Y';

numInputPoints = size(train_X,1);
[optimised_params, latent_f_opt, L, W, ~] = GPC_paramsOptimisation(params, change_vars, numSamples,num_dims, numInputPoints,train_X,train_Y,sig_func);
optimised_params_all = [optimised_params_all; optimised_params];

[~, ~, ~, ~, pi_star_ave] = GPC_inference(train_X, train_Y, optimised_params, test_X, latent_f_opt, L, W, num_dims,sig_func);
[X_AUC,Y_AUC,~,AUC] = perfcurve(test_Y,pi_star_ave,'1');
AUC
AUC_all = [AUC_all AUC];
[test_X, ~, ~, ~, pi_star_ave] = GPC_inference(train_X, train_Y, optimised_params, train_X, latent_f_opt, L, W, num_dims,sig_func);
[trainingX_AUC,trainingY_AUC,T,training_AUC] = perfcurve(train_Y,pi_star_ave,'1'); 
training_AUC
training_AUC_all = [training_AUC_all training_AUC];
end

%[sum_AUC_max, max_index] = max(AUC_all*num_test + training_AUC_all*(train_length - num_test));
[sum_AUC_max, max_index] = max(AUC_all + training_AUC_all);
sum_AUC_max
optimised_params = optimised_params_all(max_index,:);
optimised_params(:)



[X_all, K, variance, pi_star, pi_star_ave] = GPC_inference(train_X, train_Y, optimised_params, X_all, latent_f_opt, L, W, num_dims,sig_func);

pi_star_ave = pi_star_ave';
xlswrite(filename2,pi_star_ave);


pi_star_ave = sort(pi_star_ave);
plot(1:length(pi_star_ave),pi_star_ave,'b+');
% h = histogram(E)
% title('Histogram of changes in automatability from 1990 to 2010')
% xlabel('change in probability of computerisation')
% ylabel('Number of occupations')





% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ROC%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  figure
%  plot(pi_star_ave,target_Y, 'b+');
%  xlabel('MC');
%  ylabel('target class');
%  title('Ave. Probability');
%  ax = gca; % current axes
%     ax.FontSize = 14;
%  hold on;
% figure
% %[tpr,fpr,thresholds,AUC] = roc(target_Y,pi_star_ave);
% [X_AUC,Y_AUC,T,AUC] = perfcurve(target_Y,pi_star_ave,'1');
% AUC_all = [AUC_all AUC];
% % plot(X_AUC,Y_AUC)
% % xlabel('False positive rate')
% % ylabel('True positive rate')
% % title('ROC for Employment Automatability Classification')
% % outputString  = sprintf('AUC = %f',AUC);
% % text(0.5,0.5,outputString,'FontSize', 16);
% % %plotroc(target_Y,pi_star_ave);
% % ax = gca; % current axes
% %     ax.FontSize = 16;
% %  hold on   
% end
% 
% AUC = mean(AUC_all)