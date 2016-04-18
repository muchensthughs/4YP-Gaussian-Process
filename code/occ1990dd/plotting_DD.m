filename4 = '/Users/muchen/Desktop/Gaussian-Process/code/data_matching.xlsx';
filename2 = '/Users/muchen/Desktop/DD_training_data/1990_all.xlsx';

close all

A = xlsread(filename2);
D = xlsread(filename4,8);
C = xlsread(filename4,9);

changes = [];
for i = 1:701,
    if ~isnan(D(i,11))
        changes = [changes;D(i,11)];
    end
end

h = histogram(changes)
title('Histogram of changes in automatability from 1990 to 2010')
xlabel('Change in automatability')
ylabel('Number of occupations')

figure
plot(D(:,10),D(:,11),'b+')
% t = 0:1/100:1;
% hold on;
% plot(t,0*t,'color','r')
% hold on;
% t = -1:1/50:1;
% plot(0.5+0*t,t,'color','r')
title('Change vs Automatability')
xlabel('Automatability in 1990')
ylabel('Changes in automatability from 1990 to 2010')

figure
plot(D(:,10),D(:,9),'b+')
title('Change vs Automatability')
xlabel('Automatability in 1990')
ylabel('Automatability in 2010')

labor_change = 10;
array = [];
for i  = 1:length(C(:,1)),
    if ~isnan(C(i,6)) && ~isnan(C(i,labor_change)),
        array = [array; C(i,6),C(i,labor_change)]; 
    end
end
    array = sortrows(array);
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/20 * ones(1,20);
    y_ave = filter(b,a,y_init);
    figure
    plot(x,y_ave,'r-')
% figure
% plot((C(:,6)),(C(:,labor_change)),'b+')

title('Labor force change vs Automatability')
xlabel('Automatability in 1990')
ylabel('Changes in labor force percentage from 1990 to 2010')

sum = A(:,2) + A(:,3) + A(:,4);
figure
plot(A(:,2)./sum,C(:,6),'b+')
hold on
    l = lsline
    set(l,'color','r','linewidth',2);
    hold on
    plot(A(:,2)./sum,C(:,4),'go')
    ylabel('Automatability in 1990')
    xlabel('Abstract tasks')


    figure
    plot(A(:,3)./sum,C(:,6),'b+')
hold on
    l = lsline
    set(l,'color','r','linewidth',2);
    plot(A(:,3)./sum,C(:,4),'go')
    ylabel('Automatability in 1990')
    xlabel('Routine tasks')
    
    
    figure
    plot(A(:,4)./sum,C(:,6),'b+')
hold on
    l = lsline
    set(l,'color','r','linewidth',2);
    plot(A(:,4)./sum,C(:,4),'go')
    ylabel('Automatability in 1990')
    xlabel('Manual tasks')
    
  
 
 %%%%%%%%%% RTI %%%%%%%%%%%%%%
 task = A;
   perc_4 = prctile(task(:,4),6);
 perc_2 = prctile(task(:,2),5);
 for i = 1:330,
     if task(i,2) < perc_2,
         task(i,2) = perc_2;
     end
     if task(i,4) < perc_4,
         task(i,4) = perc_4;
     end
    task(i,1) = log(task(i,3)) - log(task(i,2)) - log(task(i,4)); 
 end
 RTI = task(:,1);
 
  figure
    plot(RTI,C(:,6),'b+')
hold on
    l = lsline
    set(l,'color','r','linewidth',2);
hold on
    plot(RTI,C(:,4),'go')
    ylabel('Automatability in 1990')
    xlabel('RTI')   
    %%%%%%%%%%%%%%%%%%%%%%%%%% skills vs. change in auto %%%%%%%%%%%%%%%%
    RTI_new = zeros(length(D(:,1)),1);
    for i = 1:length(D(:,1))
        if D(i,13) < perc_2,
            D(i,13) = perc_2;
        end
        if D(i,15) < perc_4,
            D(i,15) = perc_4;
        end
        RTI_new(i,1) = log(D(i,14)) - log(D(i,13)) - log(D(i,15));
    end
    
    figure 
    plot(RTI_new, D(:,11),'b+')
    hold on
    l = lsline
    xlabel('RTI')
ylabel('Changes in automatability from 1990 to 2010')
    
sum_D = D(:,13) + D(:,14) + D(:,15);    
    figure
plot(D(:,13)./sum_D,D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Abstract task score')
ylabel('Changes in automatability from 1990 to 2010')

figure
plot(D(:,14)./sum_D,D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Routine task score')
ylabel('Changes in automatability from 1990 to 2010')

figure
plot(D(:,15)./sum_D,D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Manual task score')
ylabel('Changes in automatability from 1990 to 2010')


%%%%%%%%%%%%%%%%%%%%%%%%%% skills vs. change in labor force %%%%%%%%%%%%%%
figure
plot(RTI,(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('RTI')
ylabel('Changes in labor force percentage from 1990 to 2010')

figure
plot(A(:,2)./sum,(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Abstract task score')
ylabel('Changes in labor force percentage from 1990 to 2010')

figure
plot((A(:,3)./sum),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Routine task score')
ylabel('Changes in labor force percentage from 1990 to 2010')

figure
plot((A(:,4)./sum),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Manual task score')
ylabel('Changes in labor force percentage from 1990 to 2010')





