function initialPlots(data, Attributes)
%% Initial Plot

pairs=[1 2;
       1 3;
       2 3];
xmin=min(data,[],1);
xmax=max(data,[],1);
%bounds=[xmin;xmax];
data=(data-xmin)./(xmax-xmin);
survived=data(data(:,end)==0,:);
died=data(data(:,end)==1,:);
for i=1:3
    x=pairs(i,1)
    y=pairs(i,2)
    subplot(3,1,i);
    plot(survived(:,x) ,survived(:,y),'.','MarkerSize',15); grid on;
    hold on;
    plot(died(:,x),died(:,y),'.','MarkerSize',15); grid on;
    hold off;
    xlabel(Attributes{x});
    ylabel(Attributes{y});
    legend('Survived', 'Died');
end