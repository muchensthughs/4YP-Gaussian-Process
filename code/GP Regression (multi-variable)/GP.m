%%%%%%%%%% Initialisation %%%%%%%%%%%%%%%

params = GP_combineParams(1, 0.2, 0.3, 0);
change_vars = [1, 0, 1, 0];
numSamples = 2;

X1 = [-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]';
X2 = [-1.5, -1, -0.5, 1, 0.5, 1.0, 1.5, 2.0]';
X = [X1 X2];
Y = [1.5, -0.8, 0.4, 2.0, 0.8,0.0, -0.5, 0]';

numInputPoints = size(X,1);
optimised_params = GP_paramsOptimisation(params, change_vars, numSamples,numInputPoints,X,Y );
optimised_params(:)

X_est1  = min(X1) + (0:(1e2-1))/1e2 * (max(X1) - min(X1));
X_est1 = X_est1';
X_est2  = min(X2) + (0:(1e2-1))/1e2 * (max(X2) - min(X2));
X_est2 = X_est2';
X_est = [X_est1 X_est2];
[X_est, Ymean_est, K, variance] = GP_inference(X, Y, optimised_params, X_est);


