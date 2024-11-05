%% Load data - Split data - Check Equal Proposion of Classes
data=load('haberman.data');
listOfClasses = [1 2];
preproc= 1;
equalSep=false;

while equalSep == false
    [trnData,validationData,tstData]=split_scale(data,preproc);
    [equalSep] = splitFreq(trnData,validationData,tstData, listOfClasses)
end

%% Scatter Partition
radius = 0.9;
fis=genfis2(trnData(:,1:end-1),trnData(:,end),radius);
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.1 0.9 1.1],[],validationData);
figure();
plot([trnError valError],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
title('ANFIS Classification with Scarter Partition');
Y=evalfis(tstData(:,1:end-1),valFis);
Y=round(Y);

for i=1:length(Y)
    if Y(i) > 2
        Y(i) = 2;
    end
end


%realPredictValues(tstData(:,end),Y)

%Plot MF
figure();
for i=1:3
    subplot(3,1,i);
    plotmf(fis, 'input', i);
end

%Confusion Matrix
figure();
C = confusionmat(tstData(:,end), Y);
confusionchart(C,{'Survived', 'Died'});


[c_matrix,Result,RefereceResult]= confusion_User.getMatrix(tstData(:,end),Y);


%% ANFIS - Grid Partition
fis=genfis1(trnData, 4, 'gaussmf', 'constant');
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.1 0.9 1.1],[],validationData);
figure();
plot([trnError valError],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
title('ANFIS Classification with Grid Partition');
Y=evalfis(tstData(:,1:end-1),valFis);
Y=round(Y);


%Plot MF
figure();
for i=1:3
    
    subplot(3,1,i);
    plotmf(fis, 'input', i);
end

%Confusion Matrix
figure();
C = confusionmat(tstData(:,end), Y);
confusionchart(C,{'Survived', 'Died'});

[c_matrix,Result,RefereceResult]= confusion_User.getMatrix(tstData(:,end),Y);









