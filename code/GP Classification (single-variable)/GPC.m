%%%%%%%%%% Initialisation %%%%%%%%%%%%%%%

params = GPC_combineParams(1, 0.3, 0);
change_vars = [1, 1, 0];
numSamples = 5;

X1 = [-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]';
%X2 = [-1.5, -1, -0.5, 1, 0.5, 1.0, 1.5, 2.0]';
%X = [X1 X2];
Y = [-1, -1, -1, 1, 1, 1, 1, 1]';

numInputPoints = size(X1,1);
[optimised_params, latent_f_opt] = GPC_paramsOptimisation(params, change_vars, numSamples,numInputPoints,X1,Y );
optimised_params(:)

X_est1  = min(X1) + (0:(1e3-1))/1e3 * (max(X1) - min(X1));
X_est1 = X_est1';
%X_est2  = min(X2) + (0:(1e2-1))/1e2 * (max(X2) - min(X2));
%X_est2 = X_est2';
%X_est = [X_est1 X_est2];
[X_est, K, variance] = GPC_inference(X1, Y, optimised_params, X_est1, latent_f_opt);


