function [YTest,YTrain,YPredTest,YPredTrain,train_index,test_index]=multi_disease_classifier(Data,disease_index,train,hidden_layers,hidden_nodes,num_features)
%This function considers a data set with samples from multiple diseases and
%does a NN based multidisease classification

%Data is the dataset, rows are peptide columns are samples

%disease_index is a binary index with as many rows as there are diseases
%and as many columns as there are samples which says which samples are part
%of which disease

%train is either a fraction of the samples to randomly pick for training or
%it is an actual training index (which samples will be used for training).
%If the latter it is 1 x S where S is the number of samples, so it is not
%disease specific but signifies all of the training samples

%hidden_layers is # of hidden layers in the NN

%hidden_nodes is the # of hidden nodes in the NN

%num_features is the number of features that selected from each disease...
%not the total number of features selected



%test values***********************************************
% Data=dlmread('E:\Datasets\Papers\Robayet1 submitted\G_Predicted Data from minus 77 norm using array seq  for Fig 2\predicted_array_1.csv');
% load('E:\Datasets\Papers\Robayet1 submitted\D_Consolodated array data with normalization\Sequence_and_Index_Arrays.mat');
% disease_index=Good_minus_77_disease_index;
% [N,S]=size(Data);
% num_features=20;%number of features used per disease when ttest is used to select features
% train=0.8;
% hidden_layers=1;
% hidden_nodes=300;
% fprintf('Read in %d samples and %d binding values\n',S,N);
%***********************************************************
goflag=true(1);%check for valid inputs
if hidden_layers>3
    goflag=false(1);
    fprintf('The number of hidden layers cannot be greater than 3\n')
end


if goflag
    [N,S]=size(Data);
    [num_disease,~]=size(disease_index);
    
    %First create a set of disease indices that are the train samples for each
    %disease and a set that are the test values for each disease.  We will also
    %use pvalues to pick the features (peptides) we will use in the classifier.
    % These peptides are chosen by comparing the training samples of each
    % disease to the training samples of all other diseases
    if isscalar(train) %train can either be a fraction or an actual index itself
        full_train_index=false(1,S); %This records the full training set used - it has all the training samples for all disease as true
        full_train_index(randperm(S,ceil(S*train)))=true(1); %this sets the right ones to true based on the fraction to train
    else
        full_train_index=train;
    end
    
    train_index=false(size(disease_index)); %this is the index for each disease that shows which samples we used for training
    test_index=true(size(disease_index)); %this is the index for each disease that shows which samples we used for testing
    feature_index=false(N,1); %this is the index of which peptides are going to be used as features
    for i_disease=1:num_disease
        train_index(i_disease,:)=full_train_index & disease_index(i_disease,:); %sets the right fraction of all samples to training
        test_index(i_disease,:)=~full_train_index & disease_index(i_disease,:); %sets the opposite samples to test
        train_index_other=full_train_index & ~disease_index(i_disease,:); %what we compare to for pvalues
        [~,pvalue]=ttest2(Data(:,train_index(i_disease,:))',Data(:,train_index_other)'); %ttest between that disease and all others
        sortp=sort(pvalue);
        ind3=pvalue<sortp(num_features);
        feature_index=feature_index|ind3'; %create index that captures top 20 features for each disease
    end
    
    numFeatures=sum(feature_index);
    fprintf('%d features selected\n',numFeatures);
    
    X=Data'; %need to reverse the samples vs. observations
    
    
    %create a file of zeros and ones which has S x 6 such that the first column
    %is 1 for HCV, the second for Dengue, etc.  So these are the target values
    %of the regression
    Y=double(disease_index');
    
    %run a NN model to train using the low CV files based on ttest features
%     fprintf('Starting Neural Network multiclass classifier\n');
    
    switch hidden_layers
        case 1
            layers = [
                featureInputLayer(numFeatures,'Normalization', 'zscore')
                fullyConnectedLayer(hidden_nodes)
                reluLayer
                
                fullyConnectedLayer(num_disease)
                regressionLayer];
        case 2
            layers = [
                featureInputLayer(numFeatures,'Normalization', 'zscore')
                fullyConnectedLayer(hidden_nodes)
                reluLayer
                fullyConnectedLayer(hidden_nodes)
                reluLayer
                fullyConnectedLayer(num_disease)
                regressionLayer];
        case 3
            layers = [
                featureInputLayer(numFeatures,'Normalization', 'zscore')
                fullyConnectedLayer(hidden_nodes)
                reluLayer
                fullyConnectedLayer(hidden_nodes)
                reluLayer
                fullyConnectedLayer(hidden_nodes)
                reluLayer
                fullyConnectedLayer(num_disease)
                regressionLayer];
    end
    
    miniBatchSize = 15;
    options = trainingOptions('adam', ...
        'MiniBatchSize',miniBatchSize, ...
        'Shuffle','every-epoch', ...
        'MaxEpochs',300, ...
        'InitialLearnRate',0.003,...
        'LearnRateDropFactor',0.9,...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropPeriod',2,...
        'Verbose',false);
    
    %     'Plots','training-progress', ...
    
    YTrain=Y(full_train_index,:);
    XTrain=X(full_train_index,feature_index);
    YTest=Y(~full_train_index,:);
    XTest=X(~full_train_index,feature_index);
    
    net = trainNetwork(XTrain,YTrain,layers,options);
    
    %Use the model to predict the values for the test samples
    YPredTrain = predict(net,XTrain,'MiniBatchSize',miniBatchSize);
    YPredTest = predict(net,XTest,'MiniBatchSize',miniBatchSize);
    
    [~,I]=max(YPredTest,[],2);
    [~,J]=max(YTest,[],2);
    num_correct=sum(I==J);
    fprintf('%d correct of %d or %1.3f\n',num_correct,length(I),sum(I==J)/length(I));
    
end



