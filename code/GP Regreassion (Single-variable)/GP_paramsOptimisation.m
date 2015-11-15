function [optimised_params] = GP_paramsOptimisation (initialParams, ind, numSamples, numPoints, X, Y)


%initial samples for optimisation
options = optimset('GradObj','on');
l_bounds = [log(0.1)  log(10)];
sigman_bounds = [log(0.01) log(3)];
sigmaf_bounds = [log(0.1) log(0.8)];
f_bounds = [0 2];


%numVars = sum(ind(:));
%inits = zeros(numSamples, numVars);
var_inits = [];

if ind(1)==1,
    l_samples = uniformSample(l_bounds, numSamples);
    var_inits = [var_inits l_samples];
end
if ind(2)==1,
    sigman_samples = uniformSample(sigman_bounds, numSamples);
    var_inits = [var_inits sigman_samples];
end
if ind(3)==1,
    sigmaf_samples = uniformSample(sigmaf_bounds, numSamples);
    var_inits = [var_inits sigmaf_samples];
end
if ind(4)==1,
    f_samples = uniformSample(f_bounds, numSamples);
    var_inits = [var_inits f_samples];
end

%var_inits = exp(var_inits);
%find the optimised set of parameters where function output of the
%GP_calcLikelihood (log marginal likelihood) is maxmised
params_matrix = [];
for i = 1:numSamples
    [variables, fval] = fminunc(@(variables) GP_calcLikelihood (variables,initialParams,ind, numPoints, X, Y), var_inits(i,:),options);
    params_matrix = [params_matrix; fval variables var_inits(i,:)];
end

params_matrix = sortrows(params_matrix);
chosen_params = params_matrix(1,2:end);


count = 1;
for i = 1:4
 if ind(i) == 1,
     optimised_params(i) = chosen_params(count);
     count = count + 1;
 else
     optimised_params(i) = initialParams(i);
 end
end


