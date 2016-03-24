function x_out = feature_select(x_in, ind)

x_out = [];
for i = 1:length(ind),
    x_out = [x_out, x_in(:,ind(i))];
end

end