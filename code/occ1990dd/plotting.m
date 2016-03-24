
filename4 = '/Users/muchen/Desktop/Gaussian-Process/code/data_matching.xlsx';
filename1 = '/Users/muchen/Desktop/DA_training_data/1990_all.xlsx';
filename2 = '/Users/muchen/Desktop/DD_training_data/1990_all.xlsx';

close all

DA = 1;
DD = 0;

if DA == 1 && DD == 0,
A = xlsread(filename1);
D = xlsread(filename4,3);
C = xlsread(filename4,4);

elseif DD == 1 && DA == 0,
A = xlsread(filename2);
D = xlsread(filename4,8);
C = xlsread(filename4,9);

else 
    error('Error in choosing data');
end


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

labor_change = 8;
array_temp = [C(:,4),C(:,labor_change)];
array = [];
for n = 1:length(C(:,4)),
    if ~isnan(C(n,labor_change)),
        array = [array ;array_temp(n,:)];
    end
end
array = sortrows(array);
x = array(:,1);
y_init = array(:,2);
a = 1;
b = 1/70 * ones(1,70);
y_ave = filter(b,a,y_init);
figure
plot(x,y_ave,'r-')
%plot((C(:,4)),(C(:,labor_change)),'b+')
% l = lsline
% set(l,'color','r','linewidth',2);
title('Labor force change vs Automatability')
xlabel('Automatability in 1990')
ylabel('Changes in labor force percentage from 1990 to 2010')

if DA == 1,
    
    array = [A(:,2)+A(:,4)+A(:,16),C(:,4)];
    figure 
    array = sortrows(array);
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/50 * ones(1,50);
    y_ave = filter(b,a,y_init);
    figure
    plot(x,y_ave,'r-')
  %  plot((A(:,2)+A(:,4)+A(:,16)),(C(:,4)),'b+');
    ylabel('Automatability in 1990')
    xlabel('Non-routine tasks')
    hold on 
   
    %plot((A(:,2)+A(:,4)+A(:,16)),C(:,2),'go')
    array = [C(:,4),A(:,3)+A(:,5)];
    figure 
    array = sortrows(array);
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/50 * ones(1,50);
    y_ave = filter(b,a,y_init);
    figure
    plot(x,y_ave,'r-')
    %plot((A(:,3)+A(:,5)),(C(:,4)),'b+');
    ylabel('Automatability in 1990')
    xlabel('Routine tasks')
    hold on
    l = lsline
    set(l,'color','r','linewidth',2);
    %plot((A(:,3)+A(:,5)),C(:,2),'go')
    
    figure 
    plot((A(:,2)),(C(:,4)),'b+');
    ylabel('Automatability in 1990')
    xlabel('Non-routine manual')
    hold on
    l = lsline
    set(l,'color','r','linewidth',2);
    %plot((A(:,2)),C(:,2),'go')
    
    figure
plot((A(:,2)),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine manual')
ylabel('Changes in labor force percentage from 1990 to 2010')
    
    figure 
    plot((A(:,3)),(C(:,4)),'b+');
    ylabel('Automatability in 1990')
    xlabel('Routine manual')
    hold on
    l = lsline
    set(l,'color','r','linewidth',2);
    %plot((A(:,3)),C(:,2),'go')
    
        figure
plot((A(:,3)),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Routine manual')
ylabel('Changes in labor force percentage from 1990 to 2010')
    
    figure 
    plot((A(:,4)),(C(:,4)),'b+');
    ylabel('Automatability in 1990')
    xlabel('Non-routine cognitive/interactive')
    hold on
    l = lsline
    set(l,'color','r','linewidth',2);
    %plot((A(:,4)),C(:,2),'go')
    
            figure
plot((A(:,4)),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine cognitive/interactive')
ylabel('Changes in labor force percentage from 1990 to 2010')
    
    figure 
    plot((A(:,5)),(C(:,4)),'b+');
    ylabel('Automatability in 1990')
    xlabel('Routine cognitive')
    hold on 
    l = lsline
    set(l,'color','r','linewidth',2);
    %plot((A(:,5)),C(:,2),'go')
    
    figure
plot((A(:,5)),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Routine cognitive')
ylabel('Changes in labor force percentage from 1990 to 2010')
    
    figure 
    plot((A(:,16)),(C(:,4)),'b+');
    ylabel('Automatability in 1990')
    xlabel('Non-routine cognitive/analytical')
    hold on
    l = lsline
    set(l,'color','r','linewidth',2);
    %plot((A(:,16)),C(:,2),'go')
    
        figure
plot((A(:,16)),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine cognitive/analytical')
ylabel('Changes in labor force percentage from 1990 to 2010')
   
%%%%%%%%%%%%%%%%%%%%%%%%%% skills vs. automatability change %%%%%%%%%%%%%%%%
figure
plot(D(:,13),D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine manual')
ylabel('Changes in automatability from 1990 to 2010')

    
figure
plot(D(:,14),D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Routine manual')
ylabel('Changes in automatability from 1990 to 2010')
    

    
figure
plot(D(:,15),D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine cognitive/interactive')
ylabel('Changes in automatability from 1990 to 2010')

    
figure
plot(D(:,16),D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Routine cognitive')
ylabel('Changes in automatability from 1990 to 2010')
    

figure
plot(D(:,17),D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine cognitive/analytical')
ylabel('Changes in automatability from 1990 to 2010')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif DD == 1,
    for i = 1:3,
    figure 
    plot(A(:,i+1),C(:,3),'b+');
    ylabel('Automatability in 1990')
    hold on
     plot(A(:,i+1),C(:,2),'go')
    end 
    
end
% figure 
% for i = 1:length(A(:,1)),
%     if A(i,5) == 1,
%         plot3(A(i,2),A(i,3),A(i,4),'r*');
%         hold on
%     elseif A(i,5) == 0,
%          plot3(A(i,2),A(i,3),A(i,4),'b*');
%          hold on
%     else
%         plot3(A(i,2),A(i,3),A(i,4),'k.');
%         hold on
%     end
% end
% grid on

%(automatability of each occ1990dd occupation is assigned to all corresponding 2010SOC occupation)


figure
a = C(:,4);
a = sort(a);
plot(1:length(a),a,'y+')
title('Automatability in 1990')
ylabel('automatability')
% 
% figure
% a = D(:,9);
% a = sort(a);
% plot(1:length(a),a)
% title('Automatability in 2010')
% ylabel('automatability')
% 
% figure
% a = D(:,4);
% a = sort(a);
% plot(1:length(a),a)
% title('Automatability in 2010(MO)')
% ylabel('automatability')