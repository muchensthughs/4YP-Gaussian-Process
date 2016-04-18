function y = cub_func(x)

    y = sin(x) + abs(x).^0.5 + 1.*(rand(size(x))-0.5);
    
    %y = sin(x) + abs(x).^0.5
    
end