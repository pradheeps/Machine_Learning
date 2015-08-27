% run svm on the data set with train and test set are separated
% <train,test>Data: N x D matrix, each row represent feature vector of an observation
% <train,test>Label: N x 1 matrix containing the label for each observation

clear; close all; clc;

% Load training data
load('kmeansdata.mat');
trainData = X_train(1:4000,:); 
trainLabel = y_train(1:4000,:);

% Extract important information
labelList = unique(trainLabel);
NClass = length(labelList);
[Ntrain D] = size(trainData);

% Load test data set
testData = X_test;
%testData = X_train(3701:4000,:);
%testLabel = double(y_train(3701:4000,:));
label = randi([1 10],1,15000);
testLabel = label'; 

%%
% #######################
% Parameter selection
% #######################
% First we randomly pick some observations from the training set for parameter selection
tmp = randperm(Ntrain);
evalIndex = tmp(1:ceil(Ntrain/2));
evalData = trainData(evalIndex,:);
evalLabel = trainLabel(evalIndex,:);

% #######################
% Automatic Cross Validation 
% Parameter selection using n-fold cross validation
% #######################
% ================================================================
% Note that the cross validation for parameter selection can use different
% number of fold. In tis example
% Ncv_param = 3 but 
% Ncv_classif = 5;
% Also note that we don't have to specify the fold for cv for parameter
% selection as the algorithm will pick observations into each fold
% randomly.
% ================================================================
optionCV.stepSize = 5;
optionCV.c = 1;
optionCV.gamma = 1/D;
optionCV.stepSize = 5;
optionCV.bestLog2c = 0;
optionCV.bestLog2g = log2(1/D);
optionCV.epsilon = 0.005;
optionCV.Nlimit = 100;
optionCV.svmCmd = '-q';
Ncv_param = 3; % Ncv-fold cross validation cross validation
[bestc, bestg, bestcv] = automaticParameterSelection(evalLabel, evalData, Ncv_param, optionCV);


%%

% #######################
% Classification using N-fold cross validation
% #######################

% train the svm model using the best parameters
%bestParam = ['-q -c ', num2str(bestc), ', -g ', num2str(bestg)]
model = ovrtrain(trainLabel, trainData, '-c 1 -g 0.00154 -b 1 -h 0');
%model = ovrtrain(trainLabel, trainData, bestParam);
% classify the test data set based on the svm model
[predict_label, accuracy, decis_values] = ovrpredict(testLabel, testData, model);
[decisValueWinner, predictedLabel] = max(decis_values,[],2);
writeLabels('my_labels.csv', predictedLabel);

%%
% #######################
% Make confusion matrix for the overall classification
% #######################
[confusionMatrixAll,orderAll] = confusionmat(testLabel,predictedLabel);
figure; imagesc(confusionMatrixAll');
xlabel('actual class label');
ylabel('predicted class label');
title(['confusion matrix for overall classification']);
% Calculate the overall accuracy from the overall predicted class label
accuracyAll = trace(confusionMatrixAll)/sum(confusionMatrixAll(:));
disp(['Total accuracy is ',num2str(accuracyAll*100),'%']);

% Compare the actual and predicted class
figure;
subplot(1,2,1); imagesc(testLabel); title('actual class');
subplot(1,2,2); imagesc(predictedLabel); title('predicted class');
