%% Plot with Clusters

function plotClusters(data, center1, sigmas1, center2, sigmas2)
Attributes={'Age','Year of Operation','Positive Nodes'};
pairs=[1 2;
       1 3;
       2 3];

xmin=min(data,[],1);
xmax=max(data,[],1);
%bounds=[xmin;xmax];
data=(data-xmin)./(xmax-xmin);
survived=data(data(:,end)==0,:);
died=data(data(:,end)==1,:);

%% Plot with Clusters
radius1=sigmas1(1);
radius2=sigmas2(1);
for i=1:3
    x=pairs(i,1);
    y=pairs(i,2);
    subplot(3,1,i);
    plot(survived(:,x) ,survived(:,y),'.','MarkerSize',15); grid on;
    hold on;
    plot(died(:,x) ,died(:,y),'.','MarkerSize',15); grid on;
    hold on;
    plot(center1(:,x),center1(:,y),'*','MarkerSize',20)
    hold on;
    plot(center2(:,x),center2(:,y),'*','MarkerSize',20)
    hold on;
    viscircles([center1(:,x) center1(:,y)],repmat(radius1,[size(center1,1) 1]));
    hold on;
    viscircles([center2(:,x) center2(:,y)],repmat(radius2,[size(center2,1) 1]));
    legend("Survived", "Died","Centers-Survived", "Centers-Died");
    xlabel(Attributes{x});
    ylabel(Attributes{y});
end
end