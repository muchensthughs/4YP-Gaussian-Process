function [X_est, K, Variance] = GPC_inference ( X, Y, params, X_est, latent_f_opt)



%  Initializations

length_X = size(X,1); num_ests = length(X_est(:,1));
K = zeros(length_X); K_est = zeros(length_X,1);

l = params(1);
sigma_f = params(2);
f = params(3);


%calculate the covariance matrix for known X and use it to compute
%parameters thta will be used in prediction
for i = 1:length_X,
    for j = 1:length_X,
        K(i,j) = GPC_covariance (X(i), X(j), l, sigma_f, f);
    end
end
fstar = zeros(num_ests,1); Variance = zeros(num_ests,1);

% parameter calculation
ti = (Y + 1)/2 ;
pii = 1./(1+exp(-latent_f_opt));
dlogpYf = ti - pii;
d2logpYf = -pii.*(1 - pii);

W = - diag(d2logpYf);
sqrtW = sqrtm(W);
B = eye(length_X) + sqrtW*K*sqrtW;
L = chol (B,'lower');



%  estimation 
    for q = 1:num_ests,
            for i = 1:length_X,
                K_est(i) = GPC_covariance (X(i),X_est(q), l,sigma_f, f);
            end
    
    % Mean of prediction
    fstar(q) = K_est' * dlogpYf;
    end




% d) plot
close all;
    figure, hold on

%plot the estimation line
    plot (X_est,fstar,'k')
   %plot the data points
    plot (X,Y ,'b+')

