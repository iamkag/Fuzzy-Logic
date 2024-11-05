%% Data Input
data=load('airfoil_self_noise.dat');
preproc=1;
[trnData,validationData,tstData]=split_scale(data,preproc);
Perf=zeros(4,4);


%% TSK_model_1
tic
fis=genfis1(trnData,2,'gbellmf','constant');


% MF Before Training
figure();
plotMFs(fis,size(trnData,2)-1);

% Training Model
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],validationData);
Ytst=evalfis(tstData(:,1:end-1),trnFis);
Ytrn=evalfis(trnData(:,1:end-1),trnFis);
Yval=evalfis(validationData(:,1:end-1),trnFis);


% MF After Training
figure();
plotMFs(trnFis,size(trnData,2)-1);

figure();
plot([trnError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error','Validation Error');
title('ANFIS Hybrid Training - Validation');

plot_response(trnData, Ytrn, "Train");
plot_response(validationData, Yval, "Validation");
plot_response(tstData, Ytst, "Test");



Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);
R2=Rsq(Ytst,tstData(:,end));
RMSE=sqrt(mse(Ytst,tstData(:,end)));
NDEI = RMSE/(std(Ytst));
NMSE = NDEI ^ 2;

Perf(1,1) = R2;
Perf(1,2) = RMSE;
Perf(1,3) = NDEI;
Perf(1,4) = NMSE;
Perf


%Error plot in test data (prediction)
predict_error = tstData(:,end) - Ytst;
figure();
plot(predict_error);
grid on;
xlabel('input');ylabel('Error');
title("TSK_model_1 Prediction Error");
toc

%% TSK_model_2
tic
fis=genfis1(trnData,3,'gbellmf','constant');


% MF Before Training
figure();
plotMFs(fis,size(trnData,2)-1);

% Training Model
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],validationData);
Ytst=evalfis(tstData(:,1:end-1),trnFis);
Ytrn=evalfis(trnData(:,1:end-1),trnFis);
Yval=evalfis(validationData(:,1:end-1),trnFis);


% MF After Training
figure();
plotMFs(trnFis,size(trnData,2)-1);

figure();
plot([trnError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error','Validation Error');
title('ANFIS Hybrid Training - Validation');

plot_response(trnData, Ytrn, "Train");
plot_response(validationData, Yval, "Validation");
plot_response(tstData, Ytst, "Test");



Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);
R2=Rsq(Ytst,tstData(:,end));
RMSE=sqrt(mse(Ytst,tstData(:,end)));
NDEI = RMSE/(std(Ytst));
NMSE = NDEI ^ 2;

Perf(2,1) = R2;
Perf(2,2) = RMSE;
Perf(2,3) = NDEI;
Perf(2,4) = NMSE;

%Error plot in test data (prediction)
predict_error = tstData(:,end) - Ytst;
figure();
plot(predict_error);
grid on;
xlabel('input');ylabel('Error');
title("TSK_model_2 Prediction Error");
toc



%% TSK_model_3
tic
fis=genfis1(trnData,2,'gbellmf','linear');


% MF Before Training
figure();
plotMFs(fis,size(trnData,2)-1);

% Training Model
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],validationData);
Ytst=evalfis(tstData(:,1:end-1),trnFis);
Ytrn=evalfis(trnData(:,1:end-1),trnFis);
Yval=evalfis(validationData(:,1:end-1),trnFis);


% MF After Training
figure();
plotMFs(trnFis,size(trnData,2)-1);

figure();
plot([trnError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error','Validation Error');
title('ANFIS Hybrid Training - Validation');

plot_response(trnData, Ytrn, "Train");
plot_response(validationData, Yval, "Validation");
plot_response(tstData, Ytst, "Test");



Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);
R2=Rsq(Ytst,tstData(:,end));
RMSE=sqrt(mse(Ytst,tstData(:,end)));
NDEI = RMSE/(std(Ytst));
NMSE = NDEI ^ 2;

Perf(3,1) = R2;
Perf(3,2) = RMSE;
Perf(3,3) = NDEI;
Perf(3,4) = NMSE;
Perf

%Error plot in test data (prediction)
predict_error = tstData(:,end) - Ytst;
figure();
plot(predict_error);
grid on;
xlabel('input');ylabel('Error');
title("TSK_model_3 Prediction Error");
toc



%% TSK_model_4
tic
fis=genfis1(trnData,3,'gbellmf','linear');


% MF Before Training
figure();
plotMFs(fis,size(trnData,2)-1);

% Training Model
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],validationData);
Ytst=evalfis(tstData(:,1:end-1),trnFis);
Ytrn=evalfis(trnData(:,1:end-1),trnFis);
Yval=evalfis(validationData(:,1:end-1),trnFis);


% MF After Training
figure();
plotMFs(trnFis,size(trnData,2)-1);

figure();
plot([trnError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error','Validation Error');
title('ANFIS Hybrid Training - Validation');

plot_response(trnData, Ytrn, "Train");
plot_response(validationData, Yval, "Validation");
plot_response(tstData, Ytst, "Test");



Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);
R2=Rsq(Ytst,tstData(:,end));
RMSE=sqrt(mse(Ytst,tstData(:,end)));
NDEI = RMSE/(std(Ytst));
NMSE = NDEI ^ 2;

Perf(4,1) = R2;
Perf(4,2) = RMSE;
Perf(4,3) = NDEI;
Perf(4,4) = NMSE;


%Error plot in test data (prediction)
predict_error = tstData(:,end) - Ytst;
figure();
plot(predict_error);
grid on;
xlabel('input');ylabel('Error');
title("TSK_model_4 Prediction Error");

toc

