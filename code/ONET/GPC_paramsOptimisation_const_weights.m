function [optimised_params, latent_f_opt, L, W, K] = GPC_paramsOptimisation_const_weights (initialParams, ind, numSamples, numDims, numPoints, X, Y)


%real value bounds
l_bounds_actual = [1 3];
sigmf_bounds_real = [50 70];
f_bounds_real = [0.1 1];

%initial samples for optimisation
options = optimset('GradObj','on');
l_bounds = [log(l_bounds_actual(1)) log(l_bounds_actual(2))];
sigmaf_bounds = [log(sigmf_bounds_real(1)) log(sigmf_bounds_real(2))];
f_bounds = [log(f_bounds_real(1)) log(f_bounds_real(2))];

%numVars = sum(ind(:));
%inits = zeros(numSamples, numVars);
%var_inits = [];

%if ind(1)==1,
%    l_samples = uniformSample(l_bounds, numSamples);
%    var_inits = [var_inits l_samples];
%end
%if ind(2)==1,
%    sigmaf_samples = uniformSample(sigmaf_bounds, numSamples);
%    var_inits = [var_inits sigmaf_samples];
%end
%if ind(3)==1,
%    f_samples = uniformSample(f_bounds, numSamples);
%    var_inits = [var_inits f_samples];
%end


%sampling over exponentiated values
var_inits = lhs_sample_const_weights(l_bounds, sigmaf_bounds, f_bounds, weight_bounds, numSamples, numDims, ind, 'need exponential' );
%take log of the samples so that the original sample values can be read
%corretly in calcLikelihood.m
var_inits = log(var_inits);
%find the optimised set of parameters where function output of the
%GP_calcLikelihood (log marginal likelihood) is maxmised
params_matrix = [];
for i = 1:numSamples
    [local_opt_vars, local_fmin, exitflag] = fminunc(@(variables) GPC_calcLikelihood (variables,initialParams, ind,numDims, numPoints, X, Y), var_inits(i,:),options);
    params_matrix = [params_matrix; local_fmin local_opt_vars var_inits(i,:) i];
end

figure
plot(params_matrix(:,end), params_matrix(:,1),'r+');

params_matrix = sortrows(params_matrix);
params_matrix(1,:)
chosen_params = params_matrix(1,2:1+length(local_opt_vars)); %exclude the local_fmin and choose the parameters only
count = 1;
for i = 1:3
 if ind(i) == 1,
     optimised_params(i) = chosen_params(count);
     count = count + 1;
 else
     optimised_params(i) = initialParams(i);
 end
end
optimised_params = [optimised_params chosen_params(count:end)];

%%using the optimised parameters to calculate the best latent f

[~, ~, latent_f_opt, L, W, K] = GPC_calcLikelihood (optimised_params,optimised_params,[0 0 0],numDims, numPoints, X, Y);

