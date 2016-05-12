
clear all
close all
filename4 = '/Users/muchen/Desktop/Gaussian-Process/code/data_matching.xlsx';
filename1 = '/Users/muchen/Desktop/DA_training_data/1990_all.xlsx';



A = xlsread(filename1);
D = xlsread(filename4,3);
C = xlsread(filename4,4);

color1 = [2/255 138/255 6/255];
color2 = [232 132 3]./255;
color3 = [159 159 159]./255;
color4 = [127 0 253]./255;
auto = 4;

changes = [];
for i = 1:701,
    if ~isnan(D(i,11))
        changes = [changes;D(i,11)];
    end
end


h = histogram(changes)
xlabel('Change in automatability')
ylabel('Number of occupations')
set(gca,'fontsize',12,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])

figure
plot(D(:,10),D(:,11),'b+')
% t = 0:1/100:1;
% hold on;
% plot(t,0*t,'color','r')
% hold on;
% t = -1:1/50:1;
% plot(0.5+0*t,t,'color','r')
title('Change vs Automatability')
xlabel('Automatability in 1980')
ylabel('Changes in automatability from 1980 to 2010')
set(gca,'fontsize',15)
 set(gcf,'position',[400,500,400,300])
  
 automatable_1980 = D(:,10)>0.5;
 automatable_1980 = sum(automatable_1980);
 automatable_2010 = D(:,9)>0.5;
 automatable_2010 = sum(automatable_2010);
 
figure
plot(D(:,10),D(:,9),'b+')
title('Change vs Automatability')
xlabel('Automatability in 1980')
ylabel('Automatability in 2010')
set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 
 
labor_change = 9;

figure
plot((C(:,auto)),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
title('Labor force change vs Automatability')
xlabel('Automatability in 1980')
ylabel('Changes in labor force percentage from 1980 to 2010')
set(gca,'fontsize',15,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 
 array = [];
 for i = 1:length(C(:,auto)),
    if ~isnan(C(i,auto)) && ~isnan(C(i,labor_change)) && abs(C(i,labor_change)) < 0.0025,
        array = [array; C(i,auto),C(i,labor_change)];
    end
 end
array(:,2) = array(:,2).*100;
 range = max(array(:,1)) - min(array(:,1));
low = []; middle = []; high = [];
for i = 1:length(array(:,1)),
    if array(i,1) < min(array(:,1))+1/3 * range,
        low = [low ; array(i,2)];
    elseif array(i,1) < min(array(:,1))+2/3 * range,
        middle = [middle ; array(i,2)];
    else
        high = [high ; array(i,2)];
    end
end

low_dist = [mean(low) var(low)]
middle_dist = [mean(middle) var(middle)]
high_dist = [mean(high) var(high)]
 mean(low)
 mean(middle)
 mean(high)
 
 figure
plot((array(:,1)),(array(:,2)),'k.','markersize',12,'color',color4)
l = lsline
set(l,'color','r','linewidth',2);
legend(l,'Least square line fit  ')
xlabel('Automatability in 1980')
ylabel({'Changes in labor force percentage';'from 1980 to 2010'})
set(gca,'fontsize',14,'fontweight','bold')
 set(gcf,'position',[400,500,500,400])
%%%%%%%%%%%%%%%%%%%%%%%%%%% 1990 auto vs. non-routine/routine%%%%%%%%%%%%%%% 
array = [];
sum = A(:,2) + A(:,3) + A(:,4) + A(:,5) + A(:,16); 
    array = [(A(:,2)+A(:,4)+A(:,16))./sum,C(:,auto)]; 
    array = sortrows(array);
    fliped = flipud(array);
    fliped(:,1) = -fliped(:,1);
    array = [fliped; array];
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/50 * ones(1,50);
    y_ave = filter(b,a,y_init);
    figure
    plot(x(485:end),y_ave(485:end),'r-','LineWidth',2)
  %  plot((A(:,2)+A(:,4)+A(:,16)),(C(:,4)),'b+');
    ylabel('Automatability in 1980')
    xlabel('Non-routine tasks')
    %hold on 
    %plot((A(:,2)+A(:,4)+A(:,16)),C(:,2),'go')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 axis([0 1 0 1])
  ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 0.5 1];
 
 array = [];
    array = [(A(:,3)+A(:,5))./sum,C(:,auto)];
    array = sortrows(array);
    fliped = flipud(array);
    fliped(:,1) = -fliped(:,1);
    array = [fliped; array];
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/50 * ones(1,50);
    y_ave = filter(b,a,y_init);
    figure
    plot(x(485:end),y_ave(485:end),'r-','LineWidth',2)
    ylabel('Automatability in 1980')
    xlabel('Routine tasks')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 axis([0 1 0 1])
 ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 0.5 1];
 
    figure
    plot((A(:,3)+A(:,5))./sum,(C(:,auto)),'b.','markersize',10);
    ylabel('Automatability in 1980')
    xlabel('Routine tasks')
    %plot((A(:,3)+A(:,5)),C(:,2),'go')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 0.5 1];
 
     figure
    plot((A(:,2)+A(:,4)+A(:,16))./sum,(C(:,auto)),'b.','markersize',10);
    ylabel('Automatability in 1980')
    xlabel('Non-routine tasks')
set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 0.5 1];
 
    %%%%%%%%%%%%%%%%%%%%%%%%%% 1990 auto vs. 5 tasks %%%%%%%%%%%%%%%%%%%%%%
    
    figure 
    plot((C(:,auto)),(A(:,2)./sum),'b.','markersize',10);
    xlabel('Automatability in 1980')
    ylabel('Non-routine manual')
set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 0.35 0.7];
    
    array = [A(:,2)./sum,C(:,auto)];
    array = sortrows(array);
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/70 * ones(1,70);
    y_ave = filter(b,a,y_init);
    figure 
    %plot((A(:,2)),(C(:,4)),'b+');
    plot(x,y_ave,'r-')
    ylabel('Automatability in 1980')
    xlabel('Non-routine manual')
%     hold on
%     l = lsline
%     set(l,'color','r','linewidth',2);
    %plot((A(:,2)),C(:,2),'go')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 
    figure 
    plot((C(:,auto)),(A(:,3)./sum),'b.','markersize',10);
    xlabel('Automatability in 1980')
    ylabel('Routine manual')
set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 0.5 1];

    array = [A(:,3)./sum,C(:,auto)];
    array = sortrows(array);
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/70 * ones(1,70);
    y_ave = filter(b,a,y_init);
    figure 
    %plot((A(:,3)),(C(:,4)),'b+');
    plot(x,y_ave,'r-')
    ylabel('Automatability in 1980')
    xlabel('Routine manual')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
%     hold on
%     l = lsline
%     set(l,'color','r','linewidth',2);
    %plot((A(:,3)),C(:,2),'go')
        figure 
    plot((C(:,auto)),(A(:,4)./sum),'b.','markersize',10);
    xlabel('Automatability in 1980')
    ylabel('Non-routine interactive')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 0.3 0.6];
 
 
    array = [A(:,4),C(:,auto)];
    array = sortrows(array);
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/70 * ones(1,70);
    y_ave = filter(b,a,y_init);
    figure 
    %plot((A(:,4)),(C(:,4)),'b+');
    plot(x,y_ave,'r-')
    ylabel('Automatability in 1980')
    xlabel('Non-routine interactive')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
%     hold on
%     l = lsline
%     set(l,'color','r','linewidth',2);
    %plot((A(:,4)),C(:,2),'go')
          figure 
    plot((C(:,auto)),(A(:,5)./sum),'b.','markersize',10);
    xlabel('Automatability in 1980')
    ylabel('Routine cognitive')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 0.5 1];
 
 
    array = [A(:,5),C(:,auto)];
    array = sortrows(array);
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/70 * ones(1,70);
    y_ave = filter(b,a,y_init);
    figure 
    %plot((A(:,5)),(C(:,4)),'b+');
    plot(x,y_ave,'r-')
    ylabel('Automatability in 1980')
    xlabel('Routine cognitive')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
%     hold on 
%     l = lsline
%     set(l,'color','r','linewidth',2);
    %plot((A(:,5)),C(:,2),'go')
    
              figure 
    plot((C(:,auto)),(A(:,16)./sum),'b.','markersize',10);
    xlabel('Automatability in 1980')
    ylabel('Non-routine analytical')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 0.35 0.7];
 
    array = [A(:,16),C(:,auto)];
    array = sortrows(array);
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/70 * ones(1,70);
    y_ave = filter(b,a,y_init);
    figure 
    %plot((A(:,16)),(C(:,4)),'b+');
    plot(x,y_ave,'r-')
    ylabel('Automatability in 1980')
    xlabel('Non-routine analytical')
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
%     hold on
%     l = lsline
%     set(l,'color','r','linewidth',2);
    %plot((A(:,16)),C(:,2),'go')
    


%%%%%%%%%%%%%%%%%%%%%%%%%% skills vs. change in labor force %%%%%%%%%%%%%%
array = [];
for i = 1:length(A(:,2)),
    if ~isnan(A(i,2)) && ~isnan(C(i,labor_change)),
        array = [array; (A(i,2)/sum(i)),C(i,labor_change)];
    end
end
%array = [(A(:,2)./sum),C(:,labor_change)];
    array = sortrows(array);
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/40 * ones(1,40);
    y_ave = filter(b,a,y_init);
    figure 
    %plot((A(:,16)),(C(:,4)),'b+');
    plot(x,y_ave,'r-')
% figure
% plot((A(:,2)./sum),(C(:,labor_change)),'b+')
% l = lsline
% set(l,'color','r','linewidth',2);
xlabel('Non-routine manual')
ylabel('Changes in labor force percentage from 1980 to 2010')


 range = max(array(:,1)) - min(array(:,1));
 range = 0.3;
low = []; middle = []; high = [];
for i = 1:length(array(:,1)),
    if array(i,1) < min(array(:,1))+1/3 * range,
        low = [low ; array(i,2)];
    elseif array(i,1) < min(array(:,1))+2/3 * range,
        middle = [middle ; array(i,2)];
    else
        high = [high ; array(i,2)];
    end
end

low_mean = mean(low)
middle_mean = mean(middle)
high_mean = mean(high)

figure
plot((A(:,2)./sum),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine manual')
ylabel('Changes in labor force percentage from 1980 to 2010')


figure
plot((A(:,3)./sum),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Routine manual')
ylabel('Changes in labor force percentage from 1980 to 2010')

figure
plot((A(:,4)./sum),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine cognitive/interactive')
ylabel('Changes in labor force percentage from 1980 to 2010')
    
figure
plot((A(:,5)./sum),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Routine cognitive')
ylabel('Changes in labor force percentage from 1980 to 2010')

figure
plot((A(:,16)./sum),(C(:,labor_change)),'b+')
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine cognitive/analytical')
ylabel('Changes in labor force percentage from 1980 to 2010')


%%%%%%%%%%%%%%%%%%%%%%%%%% skills vs. change in auto %%%%%%%%%%%%%%%%
sum2 = D(:,13)+D(:,14)+D(:,15)+D(:,16)+D(:,17);
figure
plot(D(:,13)./sum2,D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine manual')
ylabel('Changes in automatability from 1980 to 2010')

    
figure
plot(D(:,14)./sum2,D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Routine manual')
ylabel('Changes in automatability from 1980 to 2010')
    

    
figure
plot(D(:,15)./sum2,D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine interactive')
ylabel('Changes in automatability from 1980 to 2010')

    
figure
plot(D(:,16)./sum2,D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Routine cognitive')
ylabel('Changes in automatability from 1980 to 2010')
    

figure
plot(D(:,17)./sum2,D(:,11),'b+');
l = lsline
set(l,'color','r','linewidth',2);
xlabel('Non-routine analytical')
ylabel('Changes in automatability from 1980 to 2010')

figure
plot((D(:,15)+D(:,17))./sum2,D(:,11),'b+');
xlabel('Non-routine tasks')
ylabel('Changes in automatability from 1980 to 2010')

figure
plot((D(:,14)+D(:,16))./sum2,D(:,11),'b+');
xlabel('Routine tasks')
ylabel('Changes in automatability from 1980 to 2010')

for i = 1:length(D(:,18)),
    if abs(D(i,18)) > 0.0025,
        D(i,18) = NaN;
    end
end
figure
scatter((D(:,14)+D(:,16))./sum2,D(:,11),15,D(:,18),'filled');
xlabel('Routine tasks')
ylabel('Changes in automatability from 1980 to 2010')

array = [(D(:,14)+D(:,16))./sum2,D(:,11)];
    array = sortrows(array);
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/70 * ones(1,70);
    y_ave = filter(b,a,y_init);
    figure 
    %plot((A(:,2)),(C(:,4)),'b+');
    plot(x,y_ave,'r-')
    xlabel('Routine tasks')
ylabel('Changes in automatability from 1980 to 2010')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   
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
a = C(:,auto);
a = sort(a);
plot(1:length(a),a,'y+')
title('Automatability in 1980')
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