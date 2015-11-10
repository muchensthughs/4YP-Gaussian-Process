function [X_est, ymean_est, K, Variance] = GP_inference ( X, Y, params, X_est)



%  Initializations
Y = Y - mean(Y);
length_X = size(X,1); num_ests = length(X_est(:,1));
K = zeros(length_X); K_est = zeros(length_X,1);

l = params(1);
sigma_n = params(2);
sigma_f = params(3);
f = params(4);

%calculate the covariance matrix for known X and use it to compute
%parameters thta will be used in prediction
for i = 1:length_X,
    for j = 1:length_X,
        K(i,j) = GP_covariance (X(i,:), X(j,:), l, sigma_n, sigma_f, f);
    end
end
ymean_est = zeros(num_ests,num_ests); Variance = zeros(num_ests,num_ests);

% parameter calculation
diags = max(1e3*eps, sigma_n^2); % beef up the diagonal if sigma_n = 0
L = chol (K + diags*eye(length_X),'lower'); 
alpha = L'\(L\Y);



%  estimation 
    for q = 1:num_ests,
        for p = 1:num_ests,
            X_est1 = X_est(p,1);
            X_est2 = X_est(q,2);
            X_est_input = [X_est1 X_est2];
            for i = 1:length_X,
                K_est(i) = GP_covariance (X(i,:),X_est_input, l, sigma_n, sigma_f, f);
            end
    
    % Mean of prediction
    ymean_est(q,p) = K_est' * alpha;
    % Variance of prediction with specific one group of input X_est
    v = L\K_est;
    y_noise = sigma_n^2; % recall f* has no noise itself (Algorithm 2.1 on p 19 of H152)
    Variance(q,p) = GP_covariance (X_est_input,X_est_input,l, sigma_n, sigma_f, f) - v'*v + y_noise;
        end
    end
%bounds = [ymean_est+1.96*sqrt(Variance) ymean_est-1.96*sqrt(Variance)]+mean(Y);
ymean_est = ymean_est+mean(Y);



% d) plot
close all;
    figure, hold on

  %  color = [1 .8 .8];
%X_est = X_est';
%fill ([X_est; flipud(X_est)], [bounds(:,1); flipud(bounds(:,2))], color, 'EdgeColor', color); %draw the error region
%[x1,x2] = meshgrid(min(min(X)):0.1:max(max(X)));
contour(X_est(:,1),X_est(:,2),ymean_est,10,'ShowText','on');
plot3(X(:,1),X(:,2),Y+mean(Y),'b+');
title('Posterior Y Plot');
xlabel('X1');
ylabel('X2');
zlabel('Y');

hold on;
figure;
contour(X_est(:,1),X_est(:,2),Variance,5,'ShowText','on');
hold on;
plot(X(:,1),X(:,2),'b+');
title('Variance Plot');
xlabel('X1');
ylabel('X2');
zlabel('Var');

%plot the estimation line
   % plot (X_est,ymean_est,'k')
   %plot the data points
   % plot (X,Y+mean(Y),'b+')

