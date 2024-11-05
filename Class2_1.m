%% Load data - Split data - Check Equal Proposion of Classes
tic
data_table=readtable('epileptic_seizure_data.csv');
preproc = 1;
data=table2array(data_table(:,2:end));

listOfClasses = [1 2 3 4 5];
equalSep=false;
while equalSep == false
    [trnData,validationData,tstData]=split_scale(data,preproc);
    [equalSep] = splitFreq(trnData,validationData,tstData, listOfClasses)
end



XData = data(:,1:end-1);
YData = data(:,end);
XtrnData = trnData(:,1:end-1);
YtrnData = trnData(:,end);
XvalData = validationData(:,1:end-1);
YvalData = validationData(:,end);
XtstData = tstData(:,1:end-1);
YtstData = tstData(:,end);

%% Grid Search

%Create grid for GridSearch
numberOfFeatures = [20 40 60];
numberOfRadius = [0.3 0.9];
AccurInit = 0.0;
gridAcc = zeros(length(numberOfRadius), length(numberOfFeatures));

[ranks, weights]=relieff(XtrnData,YtrnData, size(data, 2)-1);

for j=1:length(numberOfRadius)
 
    for i=1:length(numberOfFeatures)
               
        idxOfFeatures = ranks(1, 1:numberOfFeatures(i)).';
        
        %Cross-Validation
        cvp = cvpartition(size(XtrnData,1),'kfold', 5); % Assigns Automatically 80% Training, 20% Testing
        numvalidsets = cvp.NumTestSets;


        XtestData = tstData(:, idxOfFeatures);
        ytestData = tstData(:,end);
        meanAcc = zeros(1, numvalidsets);

        for t=1:numvalidsets
            fprintf('\n Number Of Features: %d', numberOfFeatures(i));
            fprintf('\n Radius Values: %d', numberOfRadius(j));
            fprintf('\n Fold: %d \n', t);
            X = XtrnData(cvp.training(t),idxOfFeatures);
            y = YtrnData(cvp.training(t),:);
            Xvalid = XtrnData(cvp.test(t),idxOfFeatures);
            yvalid = YtrnData(cvp.test(t),:);
            
            
            Ctrain = horzcat(X, y);
            Cvalid = horzcat(Xvalid, yvalid);
            
            
            fis=genfis2(X, y, numberOfRadius(j));
            anfis_opt = anfisOptions('InitialFIS', fis, 'EpochNumber', 100, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [Xvalid yvalid]);
            [trnFis,trnError,~,valFis,valError]=anfis(Ctrain, anfis_opt);
            Y=evalfis(XtestData, valFis);
            Y=round(Y);
  
            for q=1:length(Y)
               if Y(q) > 5
                   Y(q) = 5;
               elseif Y(q) < 0.5
                   Y(q) = 1;
               end
            end
            diff=ytestData-Y;

            Acc=(length(diff)-nnz(diff))/length(Y)*100;
            meanAcc(t) = Acc;

        end
        gridAcc(j,i) = sum(meanAcc) / numvalidsets;
    end    
end

%% Final Model
tic
maxAcc = max(gridAcc, [], 'all');
radiusIdx = 0;
featureIdx= 0;
for i=1:size(gridAcc, 1)
    for j =1:size(gridAcc,2)
        if maxAcc == gridAcc(i,j)
            radiusIdx = i;
            featureIdx = j;
        end
    end
end


idxOfFeatures = ranks(1, 1:numberOfFeatures(featureIdx)).';
radius = numberOfRadius(radiusIdx);

X = trnData(:,idxOfFeatures);
y = trnData(:,end);
Xvalid = validationData(:,idxOfFeatures);
yvalid = validationData(:,end);

Ctrain = horzcat(X, y);
Cvalid = horzcat(Xvalid, yvalid);

Xtest = tstData(:, idxOfFeatures);
ytest = tstData(:, end);

Ctst = horzcat(Xtest, ytest);
 
fis=genfis2(X, y, radius);
figure();
subplot(3, 1, 1);
plotmf(fis, 'input', 1); grid on;
subplot(3,1,2);
plotmf(fis, 'input', 2); grid on;
subplot(3,1,3);
plotmf(fis, 'input', 3); grid on;



anfis_opt = anfisOptions('InitialFIS', fis, 'EpochNumber', 100, 'DisplayANFISInformation', 1, 'DisplayErrorValues', 1, 'DisplayStepSize', 1, 'DisplayFinalResults', 1, 'ValidationData', Cvalid);
[trnFis,trnError,~,valFis,valError]=anfis(Ctrain, anfis_opt);
Y=evalfis(Xtest,valFis);
figure();
subplot(3, 1, 1);
plotmf(trnFis, 'input', 1); grid on;
subplot(3,1,2);
plotmf(trnFis, 'input', 2); grid on;
subplot(3,1,3);
plotmf(trnFis, 'input', 3); grid on;



for q=1:length(Y)
    if Y(q) > 5
        Y(q) = 5;
    elseif Y(q) < 0.5
        Y(q) = 1;
    end
end
Y = round(Y);

figure();
plot([trnError valError],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
title(strcat('ANFIS Classification at Radius', num2str(radius)));
    
%Confusion Matrix
figure();
C = confusionmat(ytest, Y);
confusionchart(C);

%Predicted Error
plot_response(Ctst, Y, "Test");
    
[c_matrix,Result,RefereceResult]= confusion_User.getMatrix(ytest,Y);
toc






%% Final Model Grid Partition
tic

fis=genfis1(Ctrain, 3, 'gaussmf', 'constant');
figure();
subplot(3, 1, 1);
plotmf(fis, 'input', 1); grid on;
subplot(3,1,2);
plotmf(fis, 'input', 2); grid on;
subplot(3,1,3);
plotmf(fis, 'input', 3); grid on;

anfis_opt = anfisOptions('InitialFIS', fis, 'EpochNumber', 100, 'DisplayANFISInformation', 1, 'DisplayErrorValues', 1, 'DisplayStepSize', 1, 'DisplayFinalResults', 1, 'ValidationData', Cvalid);
[trnFis,trnError,~,valFis,valError]=anfis(Ctrain, anfis_opt);


Y=evalfis(Xtest,valFis);
figure();
subplot(3, 1, 1);
plotmf(trnFis, 'input', 1); grid on;
subplot(3,1,2);
plotmf(trnFis, 'input', 2); grid on;
subplot(3,1,3);
plotmf(trnFis, 'input', 3); grid on;

for q=1:length(Y)
    if Y(q) > 5
        Y(q) = 5;
    elseif Y(q) < 0.5
        Y(q) = 1;
    end
end
Y = round(Y);

figure();
plot([trnError valError],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
title(strcat('ANFIS Classification with MF=', num2str(3)));
    
%Confusion Matrix
figure();
C = confusionmat(ytest, Y);
confusionchart(C);
    
[c_matrix,Result,RefereceResult]= confusion_User.getMatrix(ytest,Y);
toc