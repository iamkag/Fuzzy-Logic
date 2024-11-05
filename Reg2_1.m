data_table=readtable('train.csv');
preproc =1;
data=table2array(data_table(:,2:end));
[trnData,validationData,tstData] = split_scale(data,preproc);
tic

XData = data(:,1:end-1);
YData = data(:,end);

XtrnData = trnData(:,1:end-1);
YtrnData = trnData(:,end);
XvalData = validationData(:,1:end-1);
YvalData = validationData(:,end);
XtstData = tstData(:,1:end-1);
YtstData = tstData(:,end);


%Create grid for GridSearch

numberOfFeatures = [20 40 60];
numberOfRadius = [0.3 0.9];
MSEInit = 1.0;
gridMSE = zeros(length(numberOfRadius), length(numberOfFeatures));

[ranks, weights]=relieff(XtrnData,YtrnData,100);

for j=1:length(numberOfRadius)
 
    for i=1:length(numberOfFeatures)
        
        idxOfFeatures = ranks(1, 1:numberOfFeatures(i)).';
        %Cross-Validation
        cvp = cvpartition(size(XtrnData,1),'kfold', 5); % Assigns Automatically 80% Training, 20% Testing
        numvalidsets = cvp.NumTestSets;

        XtestData = tstData(:, idxOfFeatures);
        ytestData = tstData(:,end);
        meanMSE = zeros(1, numvalidsets);
 
        for t=1:numvalidsets
            
            fprintf('\n *** Number of features: %d', numberOfFeatures(i));
            fprintf('\n *** Radii value: %d', numberOfRadius(j));
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
            
            Y=evalfis(XtestData,valFis);
            MSE=mse(Y,ytestData);


            meanMSE(t) = MSE;
            
        end
        gridMSE(j, i) = sum(meanMSE) / numvalidsets; 
    end

end


%% Final Model
minMSE = min(gridMSE, [], 2);
minMSEAll = min(minMSE);

for i=1:size(gridMSE, 1)
    for j=1:size(gridMSE, 2)
        if gridMSE(i, j) == minMSEAll
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
Xtest = tstData(:, idxOfFeatures);
ytest = tstData(:,end);

Ctrain = horzcat(X, y);
Cvalid = horzcat(Xvalid, yvalid);
Ctest = horzcat(Xtest, ytest);

 % MF Before Training
figure();
plotMFs(fis, 3);
    
fis=genfis2(X, y, radius);
anfis_opt = anfisOptions('InitialFIS', fis, 'EpochNumber', 100, 'DisplayANFISInformation', 1, 'DisplayErrorValues', 1, 'DisplayStepSize', 1, 'DisplayFinalResults', 1, 'ValidationData', [Xvalid yvalid]);
[trnFis,trnError,~,valFis,valError]=anfis(Ctrain, anfis_opt);
Y=evalfis(Xtest,valFis);
    
% MF After Training
figure();
plotMFs(trnFis,3);
    
%Responce
plot_response(Ctest, Y, "Test");

%Prediction Error
predict_error = ytest - Y;
figure();
plot(predict_error);
grid on;
xlabel('input');ylabel('Error');
title("rediction Error");

figure();
plot([trnError valError],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
title(strcat('ANFIS Classification at Radius', num2str(radius)));
    
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);
R2=Rsq(Y, ytest);
RMSE=sqrt(mse(Y, ytest));
NDEI = RMSE/(std(Y));
NMSE = NDEI ^ 2;

Perf(2,1) = R2;
Perf(2,2) = RMSE;
Perf(2,3) = NDEI;
Perf(2,4) = NMSE;
toc



%% Final Model with Grid Partition
minMSE = min(gridMSE, [], 2);
minMSEAll = min(minMSE);

for i=1:size(gridMSE, 1)
    for j=1:size(gridMSE, 2)
        if gridMSE(i, j) == minMSEAll
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
Xtest = tstData(:, idxOfFeatures);
ytest = tstData(:,end);

Ctrain = horzcat(X, y);
Cvalid = horzcat(Xvalid, yvalid);
Ctest = horzcat(Xtest, ytest);

 % MF Before Training
figure();
plotMFs(fis, 3);
    
fis=genfis1(Ctrain, 3, 'gaussmf', 'constant');
anfis_opt = anfisOptions('InitialFIS', fis, 'EpochNumber', 100, 'DisplayANFISInformation', 1, 'DisplayErrorValues', 1, 'DisplayStepSize', 1, 'DisplayFinalResults', 1, 'ValidationData', [Xvalid yvalid]);
[trnFis,trnError,~,valFis,valError]=anfis(Ctrain, anfis_opt);
Y=evalfis(Xtest,valFis);
    
% MF After Training
figure();
plotMFs(trnFis,3);
    
%Responce
plot_response(Ctest, Y, "Test");

%Prediction Error
predict_error = ytest - Y;
figure();
plot(predict_error);
grid on;
xlabel('input');ylabel('Error');
title("rediction Error");

figure();
plot([trnError valError],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
title(strcat('ANFIS Classification at Radius', num2str(radius)));
    
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);
R2=Rsq(Y, ytest);
RMSE=sqrt(mse(Y, ytest));
NDEI = RMSE/(std(Y));
NMSE = NDEI ^ 2;

Perf(2,1) = R2;
Perf(2,2) = RMSE;
Perf(2,3) = NDEI;
Perf(2,4) = NMSE;
toc
