
function [fval, gradient, latent_f] = GPC_calcLikelihood (variables,initialParams,ind, numPoints, X, Y)


% loading the parameters

% Variables:  1xnumVars containing all the variables that need to be
% opimised for each group of variable sample

varCount = 1;

if ind(1) == 1,
    l = variables(varCount);
    varCount = varCount + 1;
else
    l = initialParams(1);
end
l = exp(l);
if ind(2) == 1,
    sigma_f = variables(varCount);
    varCount = varCount + 1;
else
    sigma_f = initialParams(2);
end
sigma_f = exp(sigma_f);
if ind(3) == 1,
    f = variables(varCount);
    varCount = varCount + 1;
else
    f = initialParams(3);
end
f = exp(f);


% covariance matrix and derivatives
K = zeros(numPoints); dKdl = zeros(numPoints); dKdf = zeros(numPoints); dKdw = zeros(numPoints);
for i = 1:numPoints,
    for j = 1:numPoints,
        [K(i,j), dKdl(i,j), dKdf(i,j), dKdw(i,j)] = GPC_covariance (X(i),X(j),l, sigma_f, f);
    end
end
K = K + 1e3*eps.*eye(numPoints);

latent_f = -1*ones(numPoints,1);
ti = (Y + 1)/2 ;
pii = 1./(1+exp(-latent_f));
d2logpYf = -pii.*(1 - pii);
dlogpYf = ti - pii;

%find objective function with current settings
logpYf = -log(1 + exp(-Y.*latent_f));
logpYf = sum(logpYf);


W = - diag(d2logpYf);
sqrtW = sqrtm(W);
B = eye(numPoints) + sqrtW*K*sqrtW;
L = chol (B,'lower');
b = W*latent_f + dlogpYf;
a = b - sqrtW*L'\(L\(sqrtW*K*b));
    
obj = logpYf - (1/2)*a'*latent_f ;


obj_grad = dlogpYf - K'*latent_f;

%check whether the objective function has approached its stationary point
while norm(obj_grad) > 1e-3,
    
    %store the value of latent_f and objective function from last iteration  
    last_latent_f = latent_f;   
    obj_last = obj;

    %update latent_f 
    W = - diag(d2logpYf);
    sqrtW = sqrtm(W);
    B = eye(numPoints) + sqrtW*K*sqrtW;
    L = chol (B,'lower');
    b = W*latent_f + dlogpYf;
    a = b - sqrtW*L'\(L\(sqrtW*K*b));
    latent_f = K*a;
    step = latent_f - last_latent_f;
  
    % objective function value with current latent_f
    logpYf = -log(1 + exp(-Y.*latent_f));
    logpYf = sum(logpYf);
    obj = logpYf - (1/2)*a'*latent_f ;
    change = obj - obj_last;
    pii = 1./(1+exp(-latent_f));
    d2logpYf = -pii.*(1 - pii);
    dlogpYf = ti - pii;
    obj_grad = dlogpYf - K'*latent_f;

    %check if objective function is increasing, if not, use a smaller step
    while change < 0,
        step = (1/2).*step;
        latent_f = last_latent_f + step;
        logpYf = -log(1 + exp(-Y.*latent_f));
        logpYf = sum(logpYf);
        obj = logpYf - (1/2)*a'*latent_f;
        change = obj - obj_last;
        pii = 1./(1+exp(-latent_f));
        d2logpYf = -pii.*(1 - pii);
        dlogpYf = ti - pii;
        obj_grad = dlogpYf - K'*latent_f;
    end
end


%update parameters with optimised latent_f that will be used 
%for estimation of marginal likelihood and its gradients 
pii = 1./(1+exp(-latent_f));
dlogpYf = ti - pii;
d2logpYf = -pii.*(1 - pii);
d3logpYf = pii.*(2.*pii - 1).*(1 - pii);

W = - diag(d2logpYf);
sqrtW = sqrtm(W);
B = eye(numPoints) + sqrtW*K*sqrtW;
L = chol (B,'lower');
b = W*latent_f + dlogpYf;
a = b - sqrtW*L'\(L\(sqrtW*K*b));


% Log marginal likelihood and its gradients w.r.t. hyperparameters
logqYX = -a'*latent_f/2 - sum(log(diag(L))) + logpYf;
R = sqrtW*L'\(L\sqrtW);
C = L\(sqrtW*K);
s2 = -diag(diag(K) - diag(C'*C))/2 * d3logpYf;

%negative marginal likelihood and its gradients
fval = -logqYX; gradient = [];
if ind(1) == 1,
    s1 = (1/2)*a'*dKdl*a - (1/2)*trace(R*dKdl);
    beta = dKdl*dlogpYf;
    s3 = beta - K*R*beta;
    dlogp_dl = s1 + s2' * s3;
    gradient = [gradient -dlogp_dl];
end
if ind(2) == 1,
        s1 = (1/2)*a'*dKdf*a - (1/2)*trace(R*dKdf);
    beta = dKdf*dlogpYf;
    s3 = beta - K*R*beta;
    dlogp_df = s1 + s2' * s3;
    gradient = [gradient -dlogp_df];
end
if ind(3) == 1,
    s1 = (1/2)*a'*dKdw*a - (1/2)*trace(R*dKdw);
    beta = dKdw*dlogpYf;
    s3 = beta - K*R*beta;
    dlogp_dw = s1 + s2' * s3;
    gradient = [gradient -dlogp_dw];
end


