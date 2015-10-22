
function [covar, dKdl, dKdsigman, dKdsigmaf, dKdf] = GP_covariance (x1, x2, l, sigman, sigmaf, f)


x1 = x1';
x2 = x2';
% Covariance and gradients
covar = sigmaf^2 * exp(-(norm(x1-x2))^2/(2*l^2));
dKdl = covar * (l^-3) * (norm(x1-x2))^2; % Differentiate (2.16) from Rasmussen and Williams (2006)
dKdsigmaf = 2*sigmaf * exp(-(norm(x1-x2))^2/(2*l^2));
if f > 0,
    covar = covar + exp(-2*sin(pi*f*norm(x1-x2))^2);
    dKdf = exp(-2*sin(pi*f*norm(x1-x2))^2) * (-4*sin(pi*f*norm(x1-x2))) * cos(pi*f*norm(x1-x2)) * f*pi*norm(x1-x2);
else
    dKdf = 0;
end
if x1==x2,
    covar = covar + sigman^2;
    dKdsigman = 2*sigman;
else
    dKdsigman = 0;
end