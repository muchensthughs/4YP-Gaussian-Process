filename = '/Users/muchen/Desktop/labor force.xlsx';
filename2 = '/Users/muchen/Desktop/labor_force_2000.csv';
filename3 = '/Users/muchen/Desktop/labor_force_2000_corr_1990.csv';
A = xlsread(filename,3);
sum = zeros(975,1);
for i = 1:975,
    for j = 1:1665,
        if A(j,1) == i,
            sum(i) = sum(i)+A(j,3);
        end      
    end
end
sum = sum./123473450;
xlswrite(filename2,sum);

sum_perctg = zeros(502,1);
B = xlsread(filename,8);
for j = 1:502,
for i  = 1:9,
    if ~isnan(B(j,i+4))
    sum_perctg(j) = sum_perctg(j) + sum(B(j,i+4));
    end
end
end
xlswrite(filename3,sum_perctg);
