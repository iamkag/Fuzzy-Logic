clear
clc
tic

%% Load data - Split data

data=load('haberman.data');
listOfClasses = [1 2];
preproc= 1;
equalSep=false;

while equalSep == false
    [trnData,validationData,tstData]=split_scale(data,preproc);
    [equalSep] = splitFreq(trnData,validationData,tstData, listOfClasses)
end
%% Clustering Per Class
radius=0.9;
[c1,sig1]=subclust(trnData(trnData(:,end)==1,:),radius);
[c2,sig2]=subclust(trnData(trnData(:,end)==2,:),radius);
num_rules=size(c1,1)+size(c2,1);

%Build FIS From Scratch
fis=newfis('FIS_SC','sugeno');

%Add Input-Output Variables
names_in={'in1','in2','in3'};
for i=1:size(trnData,2)-1
    fis=addvar(fis,'input',names_in{i},[0 1]);
end
fis=addvar(fis,'output','out1',[1 2]);

%Add Input Membership Functions
name1='survived';
name2='dead';
for i=1:size(trnData,2)-1
    for j=1:size(c1,1)
        fis=addmf(fis,'input',i, strcat(name1,int2str(j)),'gaussmf',[sig1(i) c1(j,i)]);
    end
    for j=1:size(c2,1)
        fis=addmf(fis,'input',i, strcat(name2,int2str(j)),'gaussmf',[sig2(i) c2(j,i)]);
    end
end

%Add Output Membership Functions
nameout='class';
params=[zeros(1,size(c1,1)) ones(1,size(c2,1))];
for i=1:num_rules
    fis=addmf(fis,'output',1,strcat(nameout,int2str(i)),'constant',params(i));
end

%Add FIS Rule Base
ruleList=zeros(num_rules,size(trnData,2));
for i=1:size(ruleList,1)
    ruleList(i,:)=i;
end
ruleList=[ruleList ones(num_rules,2)];
fis=addrule(fis,ruleList);

%Train & Evaluate ANFIS
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],validationData);
figure();
plot([trnError valError],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
Y=evalfis(tstData(:,1:end-1),valFis);
Y=round(Y);

%Confusion Matrix
figure();
C = confusionmat(tstData(:,end), Y);
confusionchart(C,{'Survived', 'Died'});

[c_matrix,Result,RefereceResult]= confusion_User.getMatrix(tstData(:,end),Y);


%% Plots
figure();
plotMFs(fis,3);
figure();
Attributes={'Age','Year of Operation','Positive Nodes'};
initialPlots(data, Attributes)
figure();
plotClusters(data, c1, sig1, c2, sig2);
plotClustersProjection(data, c1, sig1, c2, sig2)

toc