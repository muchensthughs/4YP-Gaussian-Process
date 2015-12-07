%%%%%%%%%% Initialisation %%%%%%%%%%%%%%%
clear all

%for k = 19:24,
%params = [1, 0.3, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
params = [-1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
%change_vars = [1, 1, 0, 1,1,1,1,1,1,1,1,1,1,1];
%change_vars = [1, 1, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
change_vars = [1, 1, 0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
numSamples =20;

filename = 'US_data_email.xls';
A = xlsread(filename);
A = [A(1:123,:);A(125:end,:)];
%A(124,5) = 20+k;
num_inst = length(A(:,3));
num_dims = 17;%17

% for k = 1:num_inst,
%     if isnan(A(k,20)),
%     if A(k,2) > 0.99,
%         A(k,20) = 1;
%     elseif A(k,2) < 0.01,
%         A(k,20) = 0;
%     end
%     end
% end

 mean_A = mean (A(:,3:19));
 for j = 1:num_dims,
     A(:,j+2) = A(:,j+2) - mean_A(1,j);
 %     var = sum(A(:,j+2).^2)/num_inst;
 %     A(:,j+2) = A(:,j+2)./sqrt(var);
 end


X=[];Y=[]; class = []; Y_01 = [];

X_est =  A(:, 3:19);
results = A(:, 2);

%%%%%%%%%%%%%PCA%%%%%%%%%%%%%%%%%%%%
%  [coeff, score, latent] = princomp(X_est);
%   varorder = cumsum(latent)./sum(latent);
%   %preserve 95% variance
%  coeffinuse = coeff(:,1:11);
%  X_est = X_est * coeffinuse;
%  
for i = 1:num_inst,
    if A(i,20) == 0 ,
        X = [X; X_est(i, :)];
        Y = [Y; -1];
        Y_01 = [Y_01; 0];
    elseif A(i,20) == 1,
        X = [X; X_est(i, :)];
        Y = [Y; 1]; 
 Y_01 = [Y_01; 1];
    end
 
%   if A(i,2) >= 0.5,
%       class = [class; 1];
%   else
%       class = [class; -1];
%   end
  %X_est = X; 
end
train_length = size(X,1);
if mod(train_length,2) ~= 0,
   X = X(1:(end-1),:);
   Y = Y(1:(end-1),:);
   Y_01 = Y_01(1:(end-1),:);
   train_length = train_length - 1;
end

 r = randsample(train_length,train_length/2);
 [rest] = setdiff((1:train_length),r,'stable');
 train_X = X(r,:);
 train_Y = Y(r,:);
 target_X = X(rest,:);
 target_Y = Y_01(rest,:);
 target_Y = target_Y';



  

%X1 = [-2.0, -1.9, -1.8, -1.7, -1.5,-1,-0.5, -0.2, -0.1, 0, 0.1, 0.3, 0.4, 2.0, 2.1, 2.2, 2.3,3, 4, 5.0, 5.1, 5.2, 5.3, 5.4]';
%X2 = [-1.5, -1, -0.5, 1, 0.5, 1.0, 1.5, 2.0]';
%X2 = [-2.0, -1.9, -1.8, -1.7, -1.5,-1,-0.5, -0.2, -0.1, 0, 0.1, 0.3, 0.4, 2.0, 2.1, 2.2, 2.3,3, 4, 5.0, 5.1, 5.2, 5.3, 5.4]';

%X = [X1 X2];
%Y = [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1]';

% numInputPoints = size(X,1);
% [optimised_params, latent_f_opt, L, W, K] = GPC_paramsOptimisation(params, change_vars, numSamples,num_dims, numInputPoints,X,Y);
numInputPoints = size(train_X,1);
[optimised_params, latent_f_opt, L, W, K] = GPC_paramsOptimisation(params, change_vars, numSamples,num_dims, numInputPoints,train_X,train_Y);

optimised_params(:)

%X_est1  = min(X1) + (0:(1e1-1))/1e1 * (max(X1) - min(X1));
%X_est1 = X_est1';
%X_est2  = min(X2) + (0:(1e1-1))/1e1 * (max(X2) - min(X2));
%X_est2 = X_est2';
%X_est = [X_est1 X_est2];


%[X_est, K, variance, pi_star, pi_star_ave] = GPC_inference(X, Y, optimised_params, X_est, latent_f_opt, L, W, K, num_dims);
[target_X, K, variance, pi_star, pi_star_ave] = GPC_inference(train_X, train_Y, optimised_params, target_X, latent_f_opt, L, W, K, num_dims);

for k = 1:35,
if (pi_star_ave(k)>0.5)
    result_Y(k) = 1;
else
    result_Y(k) = 0;
   
end
end
%close all;
% figure
% plot( pi_star,results,'b+');
% xlabel('MC');
% ylabel('MO');
% title ( 'MAP Results comparison for Posterior of Probability');
% 
% %close all;
% figure
% plot( pi_star_ave,results,'b+');
% xlabel('MC');
% ylabel('MO');
% title ( 'Ave Results comparison for Posterior of Probability');
% 
% figure
% plot(pi_star,class, 'b+');
% xlabel('MC');
% ylabel('Class');
% title('Ave. Probability');

 figure
 plot(pi_star_ave,target_Y, 'b+');
 xlabel('MC');
 ylabel('target class');
 title('Ave. Probability');

[tpr,fpr,thresholds] = roc(target_Y,pi_star_ave);
plotroc(target_Y,pi_star_ave);