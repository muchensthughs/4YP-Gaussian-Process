
filename4 = '/Users/muchen/Desktop/Gaussian-Process/code/data_matching.xlsx';

D = xlsread(filename4,4);
E = [];
for i = 1:701,
    if ~isnan(D(i,9))
        E = [E;D(i,9)];
    end
end

h = histogram(E)
title('Histogram of changes in automatability from 1990 to 2010')
xlabel('change in probability of computerisation')
ylabel('Number of occupations')


figure
plot(D(:,8),D(:,9),'b+')
title('Change vs automatability')
xlabel('Automatability in 1990')
ylabel('changes in automatability from 1990 to 2010')

%(automatability of each occ1990dd occupation is assigned to all corresponding 2010SOC occupation)

figure
plot(D(:,8),D(:,10),'b+')
title('change as percentage of 2010 vs automatability')
xlabel('Automatability in 1990')
ylabel('change as percentage of 2010 in automatability from 1990 to 2010')

figure
a = D(:,8);
a = sort(a);
plot(1:length(a),a)
title('Automatability in 1990')
ylabel('automatability')

figure
a = D(:,7);
a = sort(a);
plot(1:length(a),a)
title('Automatability in 2010')
ylabel('automatability')

figure
a = D(:,4);
a = sort(a);
plot(1:length(a),a)
title('Automatability in 2010(MO)')
ylabel('automatability')