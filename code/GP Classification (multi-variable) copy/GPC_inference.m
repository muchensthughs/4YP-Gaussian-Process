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

fstar = zeros(num_ests,num_ests); Variance = zeros(num_ests,num_ests);

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

  
    for p = 1:num_ests,
        
            for i = 1:length_X,
                K_est(i) = GPC_covariance (X(i,:),X_est(p,:), l,sigma_f, w);
            end
    
    % Mean of prediction
    fstar(p) = K_est' * dlogpYf;
    %variance of prediction
    v = L\(sqrtW*K_est);
    var(p) = GPC_covariance (X_est(p,:), X_est(p,:),l, sigma_f, w) - v'*v;
sigma = sqrt(var(p));
%syms f;
%normf = normpdf(f, fstar(q,p)-1.96*sigma, fstar(q,p)+1.96*sigma);
%pi_star_ave(p) = int(  (1/(1+exp(-f)))* (1/(sigma*sqrt(2*pi))) * exp(-(f-fstar(p))^2/(2*sigma^2)) , fstar(p)-1.96*sigma, fstar(p)+1.96*sigma );
%pi_star_ave(q,p) = int( normf , fstar(q,p)-1.96*sigma, fstar(q,p)+1.96*sigma );
 
     
    end
    
pi_star = 1./(1+exp(-fstar));
%var = var(:);
%sigma = sigma(:);
%bounds = [fstar+1.96.*sqrt(var) fstar-1.96.*sqrt(var)];
 

%  plot
close all;

%    figure
%  color = [1 .8 .8];   
%fill ([X_est; flipud(X_est)], [bounds(:,1); flipud(bounds(:,2))], color, 'EdgeColor', color); %draw the error region
%contour(X_est(:,1),X_est(:,2),fstar,10,'ShowText','on');
%plot3 (X(:,1),X(:,2),ti ,'b+')
%xlabel('X1');
%ylabel('X2');
%zlabel('f');
%title('latent variable predictions');

%    figure 
%    contour(X_est(:,1),X_est(:,2),pi_star_ave,10,'ShowText','on');
%    hold on
%   %data points
%    plot3 (X(:,1),X(:,2),ti ,'b+')
%    xlabel('X1');
%    ylabel('X2');
%    zlabel('Pi');
%    title('Averaged probability');%%
%
 %   figure
 %   contour(X_est(:,1),X_est(:,2),pi_star,10,'ShowText','on');
 %   hold on
 %   plot3 (X(:,1),X(:,2),ti ,'b+')
 %   xlabel('X1');
 %   ylabel('X2');
 %   zlabel('sigma(f)');
 %   title('MAP estimation');
