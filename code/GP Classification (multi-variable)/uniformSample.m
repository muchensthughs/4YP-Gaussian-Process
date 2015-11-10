function sampled = uniformSample(bounds, N)

xmin = bounds(1);
xmax = bounds(2);

sampled = zeros(N,1);
for n = 1:N
   sampled(n) = xmin + ((xmax - xmin)/(N-1))*(n-1) ;
end
