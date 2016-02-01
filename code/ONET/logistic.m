function [val] = logistic (y, f)

val = 1/(1 + exp(-f*y));