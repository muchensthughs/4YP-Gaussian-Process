function []  =  plot_Psi( Y, count, a_mat, step, K )    


ti = (Y + 1)/2 ;

    for n = 1:count-1  
    %x = -5:0.1:5;
    %latent_f_test = latent_f;
    %a_test = last_a;
   x = -20:0.1:0;
    for i = 0:200
    %latent_f_test(n) = x(i+1);
     %a_test(n) = x(i+1);
    a_test = a_mat(:,n) + x(i+1)*step(:,n);
    latent_f_test = K*a_test;
    
    %%with correction!!!!!!!!!!!!!
%    yf = Y.*latent_f_test; s = -yf;
%    ps   = max(0,s); 
%   logpYf = -(ps+log(exp(-ps)+exp(s-ps))); 
%    logpYf = sum(logpYf);
%    s   = min(0,latent_f_test); 
%   p   = exp(s)./(exp(s)+exp(s-f));                    % p = 1./(1+exp(-f))
%    dlogpYf = (Y+1)/2-p;                          % derivative of log likelihood                         % 2nd derivative of log likelihood
%    d2logpYf = -exp(2*s-latent_f_test)./(exp(s)+exp(s-latent_f_test)).^2;
%    obj = logpYf - (1/2)*a_test'*latent_f_test ;
    
    % without corrections!!!!!!!!
    logpYf = -log(1 + exp(-Y.*latent_f_test));
    logpYf = sum(logpYf);
    obj = logpYf - (1/2)*a_test'*latent_f_test ;
    %change = obj - obj_mat(count - 1);
    pii = 1./(1+exp(-latent_f_test));
    d2logpYf = -pii.*(1 - pii);
    dlogpYf = ti - pii;
    obj_grad = dlogpYf - K'*latent_f_test;
    
    y(i+1) = obj;
    end
     close all;
    figure
    plot(x,y);
    
    end