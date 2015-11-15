
function [covar, dKdl, dKdsigmaf, dKdf] = GPC_covariance (x1,x2, l, sigmaf, f)


x1 = x1';
x2 = x2';
% Covariance and gradients
weight = ones(17,1);
d_sq = sqrd_distance(x1, x2, weight);
d = sqrt(d_sq);
covar = sigmaf^2 * exp(-d_sq/(2*l^2));
dKdl = covar * (l^-3) * d_sq; % Differentiate (2.16) from Rasmussen and Williams (2006)
dKdsigmaf = 2*sigmaf * exp(-d_sq/(2*l^2));
if f > 0,
    covar = covar + exp(-2*sin(pi*f*d)^2);
    dKdf = exp(-2*sin(pi*f*d)^2) * (-4*sin(pi*f*d)) * cos(pi*f*d) * f*pi*d;
else
    dKdf = 0;
end
