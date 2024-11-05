function plotClustersProjection(data, center1, sigmas1, center2, sigmas2)
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


%Obtain Membership functions
i=0:0.01:1;

%Plot
for p=1:size(pairs,1)
    figure();
    x=pairs(p,1);
    y=pairs(p,2);
    
    h1=[];
    v1=[];
    h2=[];
    v2=[];
    for j=1:size(center1,1)
        h1=[h1 gaussmf(i,[radius1 center1(j,x)])'];
        v1=[v1 gaussmf(i,[radius1 center1(j,y)])'];
    end
    for j=1:size(center2,1)
        h2=[h2 gaussmf(i,[radius2 center2(j,x)])'];
        v2=[v2 gaussmf(i,[radius2 center2(j,y)])'];
    end
    
    subplot(3,3,[2 3 5 6])
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
    
    subplot(3,3,[1 4]);
    plot(v1,i,'LineWidth',2);
    grid on;
    hold on;
    plot(v2, i, 'LineWidth',2);
    ylabel('x');
    xlabel('μ');
    title('Membership Functions');
    
    subplot(3,3,[8 9]);
    plot(i,h1,'LineWidth',2);
    grid on;
    hold on;
    plot(i,h2,'LineWidth',2);
    xlabel('x');
    ylabel('μ');
    title('Membership Functions');
end
end
