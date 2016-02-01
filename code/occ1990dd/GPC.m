%%%%%%%%%% Initialisation %%%%%%%%%%%%%%%
clear all




filename1 = 'ONET.xlsx';
A = xlsread(filename1,1);



filename2 = '/Users/muchen/Desktop/Gaussian-Process/code/1990results.csv';


filename3 = 'task_data.xlsx';
C = xlsread(filename3);

filename4 = '/Users/muchen/Desktop/Gaussian-Process/code/data_matching.xlsx';
D = xlsread(filename4,3);
% C(:,6) = NaN;
% requestedRow_C = [];
% for i = 1:length(B(:,1)),
%     if ~isnan(B(i,1)),
%         a = find(C(:,2) == B(i,1));
%         b = B(i,2);
%         C(a,6) = b;
%         requestedRow_C = [requestedRow_C;a];
%        
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%For C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_inst = size(C,1);
num_dims = 3;%17
numSamples = 20;

weights = zeros(1,num_dims);
change_weights = ones(1,num_dims);
params = [-1, 0.3, 0, weights];
change_vars = [1, 1, 0, change_weights];


 mean_C = mean (C(:,3:5));
 for j = 1:num_dims,
     C(:,j+2) = C(:,j+2) - mean_C(j);
%      var = sum(C(:,j+2).^2)/num_inst;
%      C(:,j+2) = C(:,j+2)./sqrt(var);
 end
 
X=[];Y=[]; class = []; Y_01 = [];
X_est =  C(:, 3:5);

%training data set
for i = 1:num_inst,
    if C(i,6) == 0 ,
        X = [X; X_est(i, :)];
        Y = [Y; -1]; %class label 1 and -1
        Y_01 = [Y_01; 0]; %class label 1 and 0
    elseif C(i,6) == 1,
        X = [X; X_est(i, :)];
        Y = [Y; 1]; 
        Y_01 = [Y_01; 1];
    end

end

train_X = X;
train_Y = Y;
target_X = X_est;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ROC%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train_length = size(X,1);
% % while mod(train_length,2) ~= 0,
% %    X = X(1:(end-1),:);
% %    Y = Y(1:(end-1),:);
% %    Y_01 = Y_01(1:(end-1),:);
% %    train_length = train_length - 1;
% % end
% 
% AUC_all = [];
% 
% for j = 1:10,
%     
%  r = randsample(train_length,train_length/2-1 );
%  [rest] = setdiff((1:train_length),r,'stable');
%  train_X = X(rest,:);
%  train_Y = Y(rest,:);
%  target_X = X(r,:);
%  target_Y = Y_01(r,:);
%  target_Y = target_Y';
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%For A%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %A(124,5) = 20+k;
% num_inst = length(A(:,3));
% num_dims = 17;%17
% 
% % for k = 1:num_inst,
% %     if isnan(A(k,20)),
% %     if A(k,2) > 0.99,
% %         A(k,20) = 1;
% %     elseif A(k,2) < 0.01,
% %         A(k,20) = 0;
% %     end
% %     end
% % end
% 
%  mean_A = mean (A(:,3:19));
%  for j = 1:num_dims,
%      A(:,j+2) = A(:,j+2) - mean_A(1,j);
%  %     var = sum(A(:,j+2).^2)/num_inst;
%  %     A(:,j+2) = A(:,j+2)./sqrt(var);
%  end
% 
% 
% X=[];Y=[]; class = []; Y_01 = [];
% 
% X_est =  A(:, 3:19);
% results = A(:, 2);
% 

% %%%%%%%%%%%%%PCA%%%%%%%%%%%%%%%%%%%%
% %  [coeff, score, latent] = princomp(X_est);
% %   varorder = cumsum(latent)./sum(latent);
% %   %preserve 95% variance
% %  coeffinuse = coeff(:,1:11);
% %  X_est = X_est * coeffinuse;
% %  
% for i = 1:num_inst,
%     if A(i,20) == 0 ,
%         X = [X; X_est(i, :)];
%         Y = [Y; -1];
%         Y_01 = [Y_01; 0];
%     elseif A(i,20) == 1,
%         X = [X; X_est(i, :)];
%         Y = [Y; 1]; 
%  Y_01 = [Y_01; 1];
%     end
%  
% %   if A(i,2) >= 0.5,
% %       class = [class; 1];
% %   else
% %       class = [class; -1];
% %   end
%   %X_est = X; 
% end
% train_length = size(X,1);
% if mod(train_length,2) ~= 0,
%    X = X(1:(end-1),:);
%    Y = Y(1:(end-1),:);
%    Y_01 = Y_01(1:(end-1),:);
%    train_length = train_length - 1;
% end
% 
%  r = randsample(train_length,train_length/2);
%  [rest] = setdiff((1:train_length),r,'stable');
%  train_X = X(r,:);
%  train_Y = Y(r,:);
%  target_X = X(rest,:);
%  target_Y = Y_01(rest,:);
%  target_Y = target_Y';
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%For A%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   

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


[target_X, K, variance, pi_star, pi_star_ave] = GPC_inference(train_X, train_Y, optimised_params, target_X, latent_f_opt, L, W, K, num_dims);

% for k = 1:train_length/9,
% if (pi_star_ave(k)>0.5)
%     result_Y(k) = 1;
% else
%     result_Y(k) = 0;
%    
% end
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pi_star_ave = pi_star_ave';
xlswrite(filename2,pi_star_ave);
D = xlsread(filename4,3);
E = [];
for i = 1:701,
    if ~isnan(D(i,9))
        E = [E;D(i,9)];
    end
end

h = histogram(E)
title('Histogram of changes in automatability from 1990 to 2010')
xlabel('change in probability of computerisation')
ylabel('Number of occupations')


%close all;
% figure
% plot( pi_star,results,'b+');
% xlabel('MC');
% ylabel('MO');
% title ( 'MAP Results comparison for Posterior of Probability');

% %close all;
% figure
% plot( pi_star_ave,results,'b+');
% xlabel('MC');
% ylabel('MO');
% title ( 'Ave Results comparison for Posterior of Probability');

% figure
% plot(pi_star_ave,class, 'b+');
% xlabel('MC');
% ylabel('Class');
% title('Ave. Probability');


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