function [d, w_inv_sum] = ave_weighted_sqrd_distance (x1, x2, w)


% calculate distance between x1 and x2 according to weight
dim = size(x1,1);
dimW = size(w,1);
%check the dimensions of input and weightings
if dim ~= dimW,
    fprintf('warning!! \n dimension of weight is not compatible with input');
if dim > dimW
    a = zeros(dim - dimW,1);
    w = [w; a];  
end

end

w_inv_sum = sum(1./(w.^2));
d = sum( (x1 - x2).^2 ./ ((w.^2).*w_inv_sum) );

