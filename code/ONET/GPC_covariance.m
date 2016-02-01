
function [covar, dKdl, dKdsigmaf, dKdf, dKdw] = GPC_covariance (x1,x2, l, sigmaf, w, weights, num_dims)


x1 = x1';
x2 = x2';
% Covariance and gradients

d_sq = sqrd_distance(x1, x2, weights);
%[d_sq_ave, w_inv_sum] = ave_weighted_sqrd_distance(x1, x2, weights);
d = sqrt(d_sq);
covar = sigmaf^2 * exp(- d_sq/(2*l^2));
dKdl = covar * (l^-3) * d_sq; % Differentiate (2.16) from Rasmussen and Williams (2006)
dKdsigmaf = 2*sigmaf * exp(-d_sq/(2*l^2));
if w > 0,
    covar = covar + exp(-2*sin(pi*w*d)^2);
    dKdf = exp(-2*sin(pi*w*d)^2) * (-4*sin(pi*w*d)) * cos(pi*w*d) * w*pi*d;
else
    dKdf = 0;
end

for i = 1:num_dims,
dKdw(i) = covar * weights(i)^(-3)*(x1(i)^2 - x2(i)^2)/(l^2);
%dKdw(i) = covar * (-1)/(l^2) * weights(i)^(-3)*( w_inv_sum^(-2)*(d_sq - weights(i)^(-2)*(x1(i)-x2(i))^2) +...
  %  (weights(i)^(-2) - 1/w_inv_sum)*((x1(i)-x2(i))^2));
end

