
function [fval, gradient] = GP_calcLikelihood (variables,initialParams,ind, numPoints, X, Y)


%%%%%%%%%%% fitting the parameters %%%%%%%%%%%%%%%%%%%%%%%%

% Variables:  1xnumVars containing all the variables that need to be
% opimised for each group of variable sample
varCount = 1;
if ind(1) == 1,
    l = variables(varCount);
    varCount = varCount + 1;
else
    l = initialParams(1);
end
exp_l = exp(l);
if ind(2) == 1,
    sigma_n = variables(varCount);
    varCount = varCount + 1;
else
    sigma_n = initialParams(2);
end
exp_sigma_n = exp(sigma_n);
if ind(3) == 1,
    sigma_f = variables(varCount);
    varCount = varCount + 1;
else
    sigma_f = initialParams(3);
end
exp_sigma_f = exp(sigma_f);
if ind(4) == 1,
    f = variables(varCount);
    varCount = varCount + 1;
else
    f = initialParams(4);
end
exp_f = exp(f);



Y = Y - mean(Y);

% covariance matrix and derivatives
K = zeros(numPoints); dKdl = zeros(numPoints); dKds = zeros(numPoints); dKdf = zeros(numPoints); dKdw = zeros(numPoints);
for i = 1:numPoints,
    for j = 1:numPoints,
        [K(i,j), dKdl(i,j), dKds(i,j), dKdf(i,j), dKdw(i,j)] = GP_covariance (X(i),X(j),exp_l, exp_sigma_n, exp_sigma_f, exp_f);
    end
end


L = chol (K,'lower');
alpha = L'\(L\Y);
invK = inv(K);
alpha2 = invK*Y; 

% Log marginal likelihood and its gradient
logpYX = -Y'*alpha/2 - sum(log(diag(L))) - numPoints*log(2*pi)/2;
dlogp_dl = exp_l*trace((alpha2*alpha2' - invK)*dKdl)/2;
dlogp_ds = exp_sigma_n*trace((alpha2*alpha2' - invK)*dKds)/2;
dlogp_df = exp_sigma_f*trace((alpha2*alpha2' - invK)*dKdf)/2;
dlogp_dw = exp_f*trace((alpha2*alpha2' - invK)*dKdw)/2;
% NB: Since l = exp(p1), dlogp/dp1 = dlogp/dl dl/dp1 = dlogp/dl exp(p1) = dlogp/dl l, where p1 = params(1)


dlogp_dl = dlogp_dl*exp_l;
dlogp_ds = dlogp_ds*exp_sigma_n;
dlogp_df = dlogp_df*exp_sigma_f;
dlogp_dw = dlogp_dw*exp_f;
%output
fval = -logpYX; gradient = [];
if ind(1) == 1,
    gradient = [gradient -dlogp_dl];
end
if ind(2) == 1,
    gradient = [gradient -dlogp_ds];
end
if ind(3) == 1,
    gradient = [gradient -dlogp_df];
end
if ind(4) == 1,
    gradient = [gradient -dlogp_dw];
end
