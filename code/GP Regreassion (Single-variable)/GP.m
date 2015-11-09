%%%%%%%%%% Initialisation %%%%%%%%%%%%%%%

params = GP_combineParams(0.3, 0.05, 0.3, 0);
change_vars = [1, 1, 0, 0];
numSamples = 5;

%X = [-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]';
%Y = [-1.5, -0.8, 0.4, 2.0, 0.8,0.0, -0.5, 0]';

X = [-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]';

Y = [-1, -1, -1, 1, 1, 1, -1, -1]';

numInputPoints = size(X,1);
optimised_params = GP_paramsOptimisation(params, change_vars, numSamples,numInputPoints,X,Y );
optimised_params(:)

X_est  = min(X) + (0:(1e3-1))/1e3 * (max(X) - min(X));
[X_est, Ymean_est, bounds, K, variance] = GP_inference(X, Y, optimised_params, X_est);