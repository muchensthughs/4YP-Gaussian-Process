function var_init = lhs_sample_const_weights(l_bounds, sigma_f_bounds, f_bounds, N, ind, needExp)

var_init = [];
num_var = sum(ind) ;
sample = lhsdesign(N,num_var);

if strcmp(needExp, 'need exponential'),
l_bounds = exp(l_bounds);
sigma_f_bounds = exp(sigma_f_bounds);
f_bounds = exp(f_bounds);
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
    f_samples = (f_bounds(2) - f_bounds(1)).*sample(:,count) + f_bounds(1);
    var_init = [var_init f_samples];
    count = count + 1;
end



