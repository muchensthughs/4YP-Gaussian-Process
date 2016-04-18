filename4 = '/Users/muchen/Desktop/2010results.xlsx';


close all



C = xlsread(filename4);
sum = zeros(length(C(:,1)),1);
for i = 1:9,
sum = sum+C(:,i+7);
end
sum = 1;
auto = 7;

figure
plot(C(:,auto),C(:,8)./sum,'b.','markersize',10)
% hold on    
%     plot(C(:,6),C(:,8)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel({'Assisting and';'caring for others'})
    set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
 ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 50 100];


    figure
    plot(C(:,auto),C(:,9)./sum,'b.','markersize',10)
% hold on
%     plot(C(:,6),C(:,9)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel('Persuasion')
    set(gca,'fontsize',17,'fontweight','bold')
     set(gcf,'position',[400,500,400,300])
      ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 40 80];
    
    figure
    plot(C(:,auto),C(:,10)./sum,'b.','markersize',10)
% hold on
%     plot(C(:,6),C(:,10)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel('Negotiation')
    set(gca,'fontsize',17,'fontweight','bold')
     set(gcf,'position',[400,500,400,300])
      ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 40 80];

     figure
    plot(C(:,auto),C(:,11)./sum,'b.','markersize',10)
% hold on
%     plot(C(:,11)./sum,C(:,6),'go')
    xlabel('Probability of computerisation ')
    ylabel('Social perceptiveness')
    set(gca,'fontsize',17,'fontweight','bold')
     set(gcf,'position',[400,500,400,300])
      ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 50 100];

       figure
    plot(C(:,auto),C(:,12)./sum,'b.','markersize',10)
% hold on
%     plot(C(:,6),C(:,12)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel('Fine arts')
    set(gca,'fontsize',17,'fontweight','bold')
     set(gcf,'position',[400,500,400,300])
      ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 50 100];

       figure
    plot(C(:,auto),C(:,13)./sum,'b.','markersize',10)
% hold on
%     plot(C(:,6),C(:,13)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel('Originality')
    set(gca,'fontsize',17,'fontweight','bold')
     set(gcf,'position',[400,500,400,300])
      ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 40 80];
     
       figure
    plot(C(:,auto),C(:,14)./sum,'b.','markersize',10)
% hold on
%     plot(C(:,6),C(:,14)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel('Manual dexterity')
    set(gca,'fontsize',17,'fontweight','bold')
     set(gcf,'position',[400,500,400,300])
      ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 40 80];

       figure
    plot(C(:,auto),C(:,15)./sum,'b.','markersize',10)
% hold on
%     plot(C(:,6),C(:,15)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel('Finger dexterity')
    set(gca,'fontsize',17,'fontweight','bold')
     set(gcf,'position',[400,500,400,300])
      ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 40 80];

       figure
    plot(C(:,auto),C(:,16)./sum,'b.','markersize',10)
% hold on
%     plot(C(:,6),C(:,16)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel('Cramped work space')
 set(gca,'fontsize',17,'fontweight','bold')
 set(gcf,'position',[400,500,400,300])
  ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 50 100];

figure
plot(C(:,auto),(C(:,8)+C(:,9)+C(:,10)+C(:,11))./sum,'b.','markersize',10)
% hold on    
%     plot(C(:,6),C(:,8)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel('Social intelligence')
    set(gca,'fontsize',17)
 set(gcf,'position',[400,500,400,300])
  ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [50 175 300];
 
 figure
plot(C(:,auto),(C(:,12)+C(:,13))./sum,'b.','markersize',10)
% hold on    
%     plot(C(:,6),C(:,8)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel('Creative intelligence')
    set(gca,'fontsize',17)
 set(gcf,'position',[400,500,400,300])
  ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 100 200];

 figure
plot(C(:,auto),(C(:,14)+C(:,15)+C(:,16))./sum,'b.','markersize',10)
% hold on    
%     plot(C(:,6),C(:,8)./sum,'go')
    xlabel('Probability of computerisation ')
    ylabel({'Perception and';' manipulation'})
    set(gca,'fontsize',17)
 set(gcf,'position',[400,500,400,300])
 ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 125 250];

array = [C(:,auto),(C(:,8)+C(:,9)+C(:,10)+C(:,11))./sum];
    array = sortrows(array);
    fliped = flipud(array);
    fliped(:,1) = -fliped(:,1);
    array = [fliped; array];
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/30 * ones(1,30);
    y_ave = filter(b,a,y_init);
    figure 
    plot(x(702:end),y_ave(702:end),'r-')
xlabel('Probability of computerisation ')
    ylabel('Social intelligence')
    set(gca,'fontsize',17)
 set(gcf,'position',[400,500,400,300])
 
 array = [C(:,auto),(C(:,12)+C(:,13))./sum];
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
    plot(x(702:end),y_ave(702:end),'r-')
xlabel('Probability of computerisation ')
    ylabel('Creative intelligence')
    set(gca,'fontsize',17)
 set(gcf,'position',[400,500,400,300])
 
 array = [C(:,auto),(C(:,14)+C(:,15)+C(:,16))./sum];
    array = sortrows(array);
    fliped = flipud(array);
    fliped(:,1) = -fliped(:,1);
    array = [fliped; array];
    x = array(:,1);
    y_init = array(:,2);
    a = 1;
    b = 1/70 * ones(1,70);
    y_ave = filter(b,a,y_init);
    figure 
    plot(x(702:end),y_ave(702:end),'r-')
xlabel('Probability of computerisation ')
    ylabel('Perception and manipulation')
    set(gca,'fontsize',17)
 set(gcf,'position',[400,500,400,300])
