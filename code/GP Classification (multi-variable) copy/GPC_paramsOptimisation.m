function [optimised_params, latent_f_opt, L, W, K] = GPC_paramsOptimisation (initialParams, ind, numSamples, numPoints, X, Y)


%initial samples for optimisation
options = optimset('GradObj','on');
l_bounds = [-5 10];
sigmaf_bounds = [0.2 0.8];
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
    params_matrix = [params_matrix; local_fmin local_opt_vars var_inits(i,:)];
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

%%using the optimised parameters to calculate the best latent f

[~, ~, latent_f_opt, L, W, K] = GPC_calcLikelihood (optimised_params,optimised_params,[0 0 0], numPoints, X, Y);

