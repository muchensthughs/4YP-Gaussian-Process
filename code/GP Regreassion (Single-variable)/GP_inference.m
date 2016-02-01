function [X_est, ymean_est, bounds, K, Variance] = GP_inference ( X, Y, params, X_est)



%  Initializations
meanY = mean(Y);
Y = Y - meanY;
length_X = size(X,1); num_ests = length(X_est);
K = zeros(length_X); K_est = zeros(length_X,1);

% hyperparameters without exponentials(exponentials are taken inside covariance function calculation)
l = params(1);
sigma_n = params(2);
sigma_f = params(3);
f = params(4);

%take exponentials for following calculation
exp_l = exp(l);
exp_sigma_n = exp(sigma_n);
exp_sigma_f = exp(sigma_f);
exp_f = exp(f);

for i = 1:length_X,
    for j = 1:length_X,
        K(i,j) = GP_covariance (X(i), X(j), exp_l, exp_sigma_n, exp_sigma_f, exp_f);
    end
end
ymean_est = zeros(num_ests,1); Variance = zeros(num_ests,1);


% calculations
diags = max(1e3*eps, exp_sigma_n^2); % beef up the diagonal if sigma_n = 0
L = chol (K + diags*eye(length_X),'lower'); 
alpha = L'\(L\Y);

%  estimation 
for q = 1:num_ests,
    for i = 1:length_X,
        K_est(i) = GP_covariance (X(i),X_est(q), exp_l, exp_sigma_n, exp_sigma_f, exp_f);
    end
    
    % Mean of prediction
    ymean_est(q) = K_est' * alpha;
    % Variance of prediction
    v = L\K_est;
    y_noise = exp_sigma_n^2; % recall f* has no noise itself (Algorithm 2.1 on p 19 of H152)
    Variance(q) = GP_covariance (X_est(q),X_est(q),exp_l, exp_sigma_n, exp_sigma_f, exp_f) - v'*v + y_noise;
%     var = Variance(q) - y_noise;
%     if var<0,
%         var
%     end
end
bounds = [ymean_est+1.96.*sqrt(Variance) ymean_est-1.96.*sqrt(Variance)]+meanY;
ymean_est = ymean_est+meanY;

% d) plot
close all;
    figure, hold on

    color = [1 .8 .8];
X_est = X_est(:);
fill ([X_est; flipud(X_est)], [bounds(:,1); flipud(bounds(:,2))], color, 'EdgeColor', color); %draw the error region


    plot (X_est,ymean_est,'k')
    plot (X,Y+meanY,'b+')
    xlabel('Input X')
    ylabel('Output Y')
    title('Posterior Mean and Variance of output Y')
    ax = gca; % current axes
    ax.FontSize = 14;
