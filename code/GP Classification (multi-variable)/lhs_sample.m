function var_init = lhs_sample(l_bounds, sigma_f_bounds, w_bounds, N, ind, needExp)

var_init = [];
num_var = sum(ind);
sample = lhsdesign(N,num_var);

if strcmp(needExp, 'need exponential'),
l_bounds = exp(l_bounds);
sigma_f_bounds = exp(sigma_f_bounds);
w_bounds = exp(w_bounds);
end

count = 1;


if ind(1) == 1,
    l_samples = (l_bounds(2) - l_bounds(1)).*sample(:,count) + l_bounds(1);
    var_init = [var_init l_samples];
    count = count + 1;
end
if ind(2) == 1,
    sigma_f_samples = (sigma_f_bounds(2) - sigma_f_bounds(1)).*sample(:,count) + sigma_f_bounds(1);
    var_init = [var_init sigma_f_samples];
    count = count + 1;
end
if ind(3) == 1,
    w_samples = (w_bounds(2) - w_bounds(1)).*sample(:,count) + w_bounds(1);
    var_init = [var_init w_samples];
    count = count + 1;
end

