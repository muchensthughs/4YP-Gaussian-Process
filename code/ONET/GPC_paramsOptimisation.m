function [optimised_params, latent_f_opt, L, W, K] = GPC_paramsOptimisation (initialParams, ind, numSamples, numDims, numPoints, X, Y)


%real value bounds
l_bounds = [0.1 10];
sigmaf_bounds= [0.1 30];
f_bounds = [0.1 1];

%weight_bounds = [1 5; 1 5; 1 5; 1 5;1 5;1 5;1 5;1 5;1 5;1 5;1 5; 1 5; 1 5; 1 5; 1 10; 1 10; 1 10];
weight_bounds = zeros(numDims,2);
weight_bounds(:,1) = 0.1;
weight_bounds(:,2) = 25;
%weight_bounds = [1 10; 1 10; 1 10; 1 10;1 10;1 10;1 10;1 10;1 10;1 10;1 10];

%weight_bounds = [1 100];
%initial samples for optimisation
options = optimoptions(@fminunc,'Algorithm','quasi-newton','GradObj','on');
l_bounds = [log(l_bounds(1)) log(l_bounds(2))];
sigmaf_bounds = [log(sigmaf_bounds(1)) log(sigmaf_bounds(2))];
f_bounds = [log(f_bounds(1)) log(f_bounds(2))];
for i=1:numDims,
 weight_bounds(i,:) = [log(weight_bounds(i,1)) log(weight_bounds(i,2))];
end
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
var_inits = lhs_sample(l_bounds, sigmaf_bounds, f_bounds, weight_bounds, numSamples, numDims, ind, 'need exponential' );
%take log of the samples so that the original sample values can be read
%corretly in calcLikelihood.m
var_inits = log(var_inits);
upperbound = []; lowerbound = [];
upperbound =[upperbound, l_bounds(2), sigmaf_bounds(2),  weight_bounds(:,2)'];
lowerbound = [lowerbound, l_bounds(1), sigmaf_bounds(1),  weight_bounds(:,1)'];
%upperbound = exp(upperbound);
%lowerbound = exp(lowerbound);
%find the optimised set of parameters where function output of the
%GP_calcLikelihood (log marginal likelihood) is maxmised
params_matrix = [];
A = [];
b = [];
Aeq = [];
beq = [];

for i = 1:numSamples
    funcObj = @(variables) GPC_calcLikelihood (variables,initialParams, ind,numDims, numPoints, X, Y);
    funcProj = @(variables)boundProject(variables,zeros([19,1]),upperbound');
    vars = var_inits(i,:);
%   [local_opt_vars, local_fmin ] = fmincon(@(variables) GPC_calcLikelihood (variables,initialParams, ind,numDims, numPoints, X, Y), var_inits(i,:),A,b,Aeq,beq,zeros([1,19]),upperbound);
   [local_opt_vars, local_fmin ] = minConf_PQN(funcObj, vars',funcProj,[]);

 %  [local_opt_vars, local_fmin, exitflag] = fminunc(@(variables) GPC_calcLikelihood (variables,initialParams, ind,numDims, numPoints, X, Y), var_inits(i,:),options);
%[local_opt_vars, local_fmin, exitflag] = fminsearch(@(variables) GPC_calcLikelihood (variables,initialParams, ind,numDims, numPoints, X, Y, const_weights), var_inits(i,:));
%[local_opt_vars, local_fmin] = patternsearch(@(variables) GPC_calcLikelihood (variables,initialParams, ind,numDims, numPoints, X, Y), var_inits(i,:),[],[],[],[],[l_bounds;sigmaf_bounds;weight_bounds]);

  params_matrix = [params_matrix; local_fmin local_opt_vars' var_inits(i,:) i];
end

% figure
% plot(params_matrix(:,end), params_matrix(:,1),'r+');

params_matrix = sortrows(params_matrix);
params_matrix(1,:)
chosen_params = params_matrix(1,2:1+length(local_opt_vars)); %exclude the local_fmin and choose the parameters only
count = 1;
for i = 1:length(ind)
 if ind(i) == 1,
     optimised_params(i) = chosen_params(count);
     count = count + 1;
 else
     optimised_params(i) = initialParams(i);
 end
end

%%using the optimised parameters to calculate the best latent f

[~, ~, latent_f_opt, L, W, K] = GPC_calcLikelihood (optimised_params,optimised_params,ind,numDims, numPoints, X, Y);

