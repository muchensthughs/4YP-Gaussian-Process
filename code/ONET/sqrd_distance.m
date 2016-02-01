function [d] = sqrd_distance (x1, x2, w)


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

d = 0;
for n = 1:dim,
    d = (x1(n) - x2(n))^2 / w(n)^2 + d;  
end
