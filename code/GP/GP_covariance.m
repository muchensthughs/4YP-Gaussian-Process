
function [covar, dKdl, dKdsigman, dKdsigmaf, dKdf] = GP_covariance (x1, x2, l, sigman, sigmaf, f)



% Covariance and gradients
covar = sigmaf^2 * exp(-(x1-x2)^2/(2*l^2));
dKdl = covar * (l^-3) * (x1-x2)^2; % Differentiate (2.16) from Rasmussen and Williams (2006)
dKdsigmaf = 2*sigmaf * exp(-(x1-x2)^2/(2*l^2));
if f > 0,
    covar = covar + exp(-2*sin(pi*f*(x1-x2))^2);
    dKdf = exp(-2*sin(pi*f*(x1-x2))^2) * (-4*sin(pi*f*(x1-x2))) * cos(pi*f*(x1-x2)) * f*pi*(x1-x2);
else
    dKdf = 0;
end
if x1==x2,
    covar = covar + sigman^2;
    dKdsigman = 2*sigman;
else
    dKdsigman = 0;
end