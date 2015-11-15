%%%%%%%%%% Initialisation %%%%%%%%%%%%%%%

params = GPC_combineParams(1, 0.3, 0);
change_vars = [1, 1, 0];
numSamples = 50;

filename = 'US_data_email.xls';
A = xlsread(filename);



X1 = [-2.0, -1.9, -1.8, -1.7, -1.5,-1,-0.5, -0.2, -0.1, 0, 0.1, 0.3, 0.4, 2.0, 2.1, 2.2, 2.3,3, 4, 5.0, 5.1, 5.2, 5.3, 5.4]';
%X2 = [-1.5, -1, -0.5, 1, 0.5, 1.0, 1.5, 2.0]';
%X = [X1 X2];
Y = [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1]';

numInputPoints = size(X1,1);
[optimised_params, latent_f_opt, L, W, K] = GPC_paramsOptimisation(params, change_vars, numSamples,numInputPoints,X1,Y );
optimised_params(:)

X_est1  = min(X1) + (0:(1e2-1))/1e2 * (max(X1) - min(X1));
X_est1 = X_est1';
%X_est2  = min(X2) + (0:(1e2-1))/1e2 * (max(X2) - min(X2));
%X_est2 = X_est2';
%X_est = [X_est1 X_est2];
[X_est, K, variance] = GPC_inference(X1, Y, optimised_params, X_est1, latent_f_opt, L, W, K);


