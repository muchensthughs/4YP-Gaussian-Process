%%%%%%%%%% Initialisation %%%%%%%%%%%%%%%

params = GP_combineParams(log(0.25), log(0.3), log(0.3), 0);
change_vars = [0, 1, 1, 0];
numSamples = 5;

%X = [-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]';
%Y = [-1.5, -0.8, 0.4, 2.0, 0.8,0.0, -0.5, 0]';

% X = [-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]';
% X_others = X+0.5;
% Y = cub_func(X);
% Y_others = cub_func(X_others);

numInputPoints = size(X,1);
optimised_params = GP_paramsOptimisation(params, change_vars, numSamples,numInputPoints,X,Y );
exp(optimised_params(:))

X_est  = min(X) + (0:(1e3-1))/1e3 * (max(X) - min(X));
[X_est, Ymean_est, bounds, K, variance] = GP_inference(X, Y, optimised_params, X_est);

% plot(X_others, Y_others,'r+')
% hold on
x_origin = 0:0.1:13;
y_origin = sin(x_origin) + abs(x_origin).^0.5;
plot(x_origin, y_origin,'g','LineWidth',2)
legend('variance','GP model','noisy samples','original function','Location','northwest')