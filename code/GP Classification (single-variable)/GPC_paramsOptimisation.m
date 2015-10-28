function [optimised_params, latent_f_opt] = GPC_paramsOptimisation (initialParams, ind, numSamples, numPoints, X, Y)


%initial samples for optimisation
options = optimset('GradObj','on');
l_bounds = [0.5 2];
sigmaf_bounds = [0.3 0.5];
f_bounds = [0 5];

%numVars = sum(ind(:));
%inits = zeros(numSamples, numVars);
var_inits = [];

if ind(1)==1,
    l_samples = uniformSample(l_bounds, numSamples);
    var_inits = [var_inits l_samples];
end
if ind(2)==1,
    sigmaf_samples = uniformSample(sigmaf_bounds, numSamples);
    var_inits = [var_inits sigmaf_samples];
end
if ind(3)==1,
    f_samples = uniformSample(f_bounds, numSamples);
    var_inits = [var_inits f_samples];
end


%find the optimised set of parameters where function output of the
%GP_calcLikelihood (log marginal likelihood) is maxmised
params_matrix = [];
for i = 1:numSamples
    [local_opt_vars, local_fmin, exitflag] = fminunc(@(variables) GPC_calcLikelihood (variables,initialParams,ind, numPoints, X, Y), var_inits(i,:),options);
    params_matrix = [params_matrix; local_fmin local_opt_vars exitflag];
end

params_matrix = sortrows(params_matrix);
chosen_params = params_matrix(1,2:(end-1)); %exclude the local_fmin and choose the parameters only
count = 1;
for i = 1:3
 if ind(i) == 1,
     optimised_params(i) = chosen_params(count);
     count = count + 1;
 else
     optimised_params(i) = initialParams(i);
 end
end

%%using the optimised parameters to calculate latent f
latent_f_opt = zeros(numPoints,1);
l = optimised_params(1);
sigma_f = optimised_params(2);
f = optimised_params(3);

% covariance matrix and derivatives
K = zeros(numPoints); dKdl = zeros(numPoints); dKdf = zeros(numPoints); dKdw = zeros(numPoints);
for i = 1:numPoints,
    for j = 1:numPoints,
        [K(i,j), dKdl(i,j), dKdf(i,j), dKdw(i,j)] = GPC_covariance (X(i),X(j),l, sigma_f, f);
    end
end
K = K + 1e3*eps*eye(numPoints);

ti = (Y + 1)/2 ;
error = 1;
while error > 1e-8,
pii = 1./(1+exp(-latent_f_opt));
d2logpYf = -pii.*(1 - pii);
dlogpYf = ti - pii;

W = - diag(d2logpYf);
sqrtW = sqrtm(W);
B = eye(numPoints) + sqrtW*K*sqrtW;
L = chol (B,'lower');
b = W*latent_f_opt + dlogpYf;
a = b - sqrtW*L'\(L\(sqrtW*K*b));

last_f = latent_f_opt;
latent_f_opt = K*a;
error = norm(latent_f_opt - last_f);
end

