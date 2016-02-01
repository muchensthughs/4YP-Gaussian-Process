function [X_est, K, Variance] = GPC_inference ( X, Y, params, X_est, latent_f_opt, L, W, K)



%  Initializations

length_X = size(X,1); num_ests = length(X_est(:,1));
K = zeros(length_X); K_est = zeros(length_X,1);

l = params(1);
sigma_f = params(2);
w = params(3);

l = exp(l);
sigma_f = exp(sigma_f);
w = exp(w);

%calculate the covariance matrix for known X and use it to compute
%parameters thta will be used in prediction
%for i = 1:length_X,
%    for j = 1:length_X,
%        K(i,j) = GPC_covariance (X(i), X(j), l, sigma_f, f);
%    end
%end

fstar = zeros(num_ests,1); Variance = zeros(num_ests,1);

% parameter calculation
ti = (Y + 1)/2 ;


%with corrections!!!!!!!!!!
yf = Y.*latent_f_opt; s = -yf;
    ps   = max(0,s); 
    logpYf = -(ps+log(exp(-ps)+exp(s-ps))); 
    logpYf = sum(logpYf);
    s   = min(0,latent_f_opt); 
    p   = exp(s)./(exp(s)+exp(s-latent_f_opt));                    % p = 1./(1+exp(-f))
    dlogpYf = ti-p;                          % derivative of log likelihood                         % 2nd derivative of log likelihood
    d2logpYf = -exp(2*s-latent_f_opt)./(exp(s)+exp(s-latent_f_opt)).^2;
    d3logpYf = 2*d2logpYf.*(0.5-p);


sqrtW = sqrt(W);
%  estimation 
    for q = 1:num_ests,
            for i = 1:length_X,
                K_est(i) = GPC_covariance (X(i),X_est(q), l,sigma_f, w);
            end
    
    % Mean of prediction
    fstar(q) = K_est' * dlogpYf;
    %variance of prediction
    v = L\(sqrtW*K_est);
    var(q) = GPC_covariance (X_est(q),X_est(q),l, sigma_f, w) - v'*v;
    if var(q)<0,
        var(q)
    end
sigma = sqrt(var(q));
syms f;
%normf = normpdf(f, fstar(q)-1.96*sigma, fstar(q)+1.96*sigma);
%pi_star_ave(q) = int(  (1/(1+exp(-f)))* (1/(sigma*sqrt(2*pi))) * exp(-(f-fstar(q))^2/(2*sigma^2)) , fstar(q)-1.96*sigma, fstar(q)+1.96*sigma );
%pi_star_ave(q) = int( normf , fstar(q)-1.96*sigma, fstar(q)+1.96*sigma );
 pi_star_ave(q) = normcdf(fstar(q)/sqrt(1+var(q)));
    end
    
pi_star = 1./(1+exp(-fstar));
var = var(:);
sigma = sigma(:);
bounds = [fstar+1.96.*sqrt(var) fstar-1.96.*sqrt(var)];
 

%  plot
close all;
    figure
  color = [1 .8 .8];   
fill ([X_est; flipud(X_est)], [bounds(:,1); flipud(bounds(:,2))], color, 'EdgeColor', color); %draw the error region
hold on
plot (X_est,fstar,'r')
plot (X,ti ,'b+')
ylabel('Latent f');
xlabel('X');
title('Posterior Mean and Variance of latent f');
ax = gca; % current axes
     ax.FontSize = 14;
     
     
     figure 
     plot (X_est,pi_star_ave,'k')
     hold on
%    %data points
     plot (X,ti ,'b+')
     ylabel('Predictive Probability');
     xlabel('X');
     title('Averaged Prediction');
     ax = gca; % current axes
     ax.FontSize = 18;

    figure 
    plot (X_est,pi_star,'k')
    hold on
    plot (X,ti ,'b+')
    ylabel('Predictive Probability');
    xlabel('X');
    title('MAP Prediction');
         ax = gca; % current axes
    ax.FontSize = 18;