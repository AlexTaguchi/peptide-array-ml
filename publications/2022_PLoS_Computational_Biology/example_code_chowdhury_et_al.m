%This Matlab code is associated with Chowdhury et al PLOS Computational Biology
%It gives an example of the major algorithms used in the paper
%
%It can be run directly from the data file that has been uploaded
%
%It starts by performing a neural network relating the sequences on the 
%   peptide array to the binding values for a set of sera samples

%The model that contains the sequence binding relationship that results from
%the neural network is then used to create three things:
%   A set of binding values PREDICTED for the sequences on the array
%   A set of binding values PROJECTED for a new set of random sequences
%   A set of vectors, one for each sample, corresponding to the final
%       weight matrix for the neural network model.
%
%Each of the three sets of values above and the original measured values
%are run through a multiclass classifier which trains on a training set and
%then tests on a test set.  It picks the train and test sets multiple times
%and averages the accuracy for each of the datasets above.  This is printed
%as a table along with the error in the mean
%
%There are six cohorts with the following names: HCV, Dengue, WNV, HBV,
%Chagas, and ND (HCV is Hepatitis C, WNV is west nile virus, HBV is
%Hep B, ND is no known disease
%
%There are three different sample types: 'BLowCV' is the balanced dataset
%of 465 samples, all of which are low CV samples and excluding 77 of the 
%low CV ND samples in order to balance the numbers better.  'NLowCV' are the
%77 excluded normals that have low CVs.  'HighCV' are the 137 samples with
%high coefficients of variation on the array of the measured control
%peptide values.  All together there are 679 samples and 122,926 peptides.
clear
close all

%% Part 1 reading in the dataset******************************************
fprintf('Reading the measured data\n')
seq=readcell('Chowdhury_et_al_dataset.csv','Range','A:A');
seq=seq(3:end,1);
Disease=readcell('Chowdhury_et_al_dataset.csv','Range','1:1');
Disease=Disease(1,2:end);%first column is blank due to sequences
sample_set=readcell('Chowdhury_et_al_dataset.csv','Range','2:2');
sample_set=sample_set(1,2:end);%first column is blank due to sequences
Data=readmatrix('Chowdhury_et_al_dataset.csv');
Data=Data(:,2:end);%first column is NAN due to sequences
disease_names={'HCV';'Dengue';'WNV';'HBV';'Chagas';'ND'};
sample_types={'BLowCV';'NLowCV';'HighCV'};
[N,S]=size(Data);

disease_index=false(length(disease_names),S);
for i=1:length(disease_names)
    disease_index(i,:)=strcmp(Disease,disease_names(i));
end

BLowCV_index=strcmp(sample_set,'BLowCV');
NLowCV_index=strcmp(sample_set,'NLowCV');
HighCV_index=strcmp(sample_set,'HighCV');

%convert the sequences to a character array
Sequence=char(N,20);
plen=zeros(N,1);
for n=1:N
    aa=seq{n};
    plen(n)=length(aa);
    Sequence(n,1:plen(n))=aa;
end
Sequence=Sequence(:,1:max(plen)); %make it the length of the longest peptide

%at this point you have:
%   Data = a matrix that has as many rows as there are peptides on the array
%   Sequence = character array, one sequence in each row
%   disease_names = cell array with the names of the six diseases
%   disease_index = 6 row logical array, each row is the binary index for a
%       different disease (rows are in the same order as disease names)
%   BLowCV_index = the binary index to the balanced samples set of low CV samples
%   NLowCV_index = the binary index to the 77 additional low CV ND samples
%   HighCV_index = the binary index to the high CV samples

%% Part 2 train the neural network*************************************
%For this example, we will train the network on 465 balanced, low CV samples
Data=Data(:,BLowCV_index);
disease_index=disease_index(:,BLowCV_index);
[N,S]=size(Data);
%**********USER SET PARAMETERS*********************************************
hidden_layers=3; %layers in NN fit
hidden_nodes=250; %Nodes in NN fit
log_shift=0.0;%I have removed zeros already
train_fraction=0.99; %training fraction of peptides used in NN fit
Max_epochs=20; %training epochs in NN fit
Batch_size=100; %number of peptides used in each training step of the NN fit
learn_rate=0.001; %size of initial optimization steps
Randomize_sequence_order=false(1); %as a control, once can scramble the order of the sequences vs. the binding values
%**************************************************************************

aminos='ADEFGHKLNPQRSVWY'; %These are the amino acids used in the arrays for this publication

Sequence_bin=seq2bin(Sequence,aminos); %converts amino acid sequences to one-hot vectors

%Convert measured data to log10 - Note data has already been median normalized for each
%sample
Data(Data<max(Data(1:end))/10000)=max(Data(1:end))/10000; %get rid of zeros and replace with a small number
Data=log10(Data); %take the log base 10

%Can randomize sequence order as a control - should have near zero
%correlation between predicted and measured outputs
if Randomize_sequence_order
    Sequence_bin=Sequence_bin(randperm(N),:);
end

%next choose a set of training peptides
numtrain=int32(train_fraction*N); %number of peptides used to train
rng('shuffle'); %results in a new seed every time
[Train_index,Test_index]=select_rand_peptides(N,numtrain); %creates index for the training peptides and test peptides

%Now run the NN fit with all the data
XTrain=Sequence_bin(Train_index,:);
YTrain=Data(Train_index,:);
XTest=Sequence_bin(~Train_index,:);
YTest=Data(~Train_index,:);

fprintf('Running the Sequence vs. Binding Neural Network\n')
%run a neural network regression using the one hot values as the input
%values and the binding values as the target

numFeatures = size(XTrain,2); %number of features in the one-hot description of the sequence
numtargets = size(YTrain,2); %number of samples being fit

%create the structure fo the neural network based on numbers of layers and
%features in the layers.  This is a fully connected, regressor
layers=featureInputLayer(numFeatures,'Normalization', 'zscore'); %set up the input layer
for i_layers=1:hidden_layers %add the right number of hidden layers and activation functions
    layers=[
        layers
        fullyConnectedLayer(hidden_nodes)
        reluLayer];
end
layers=[layers
    fullyConnectedLayer(numtargets)
    regressionLayer]; %add the regression layer to the end

%Set up the options for the training.  Note the validation is done on the
%test set
options = trainingOptions('adam', ...
    'MiniBatchSize',Batch_size, ...
    'Shuffle','every-epoch', ...
    'MaxEpochs',Max_epochs, ...
    'InitialLearnRate',learn_rate,...
    'LearnRateDropFactor',0.9,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',2,...
    'ValidationData',{XTest,YTest}, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'Verbose',false);

%Add or remove this from options to add or remove the plotting during the
%run.  It runs considerably faster without the plot and without the
%validation data
% 'Plots','training-progress', ...

%To use the CPU rather than the GPU add the following to options.  Defaults to a GPU
%'ExecutionEnvironment','cpu',...

%Run the neural network training
tic %start timer
[NN_model,info]= trainNetwork(XTrain,YTrain,layers,options);
toc %report fit time

%Predict values of the array sequences

%Generally, it is faster to do the
%prediction in chunks like this (you can also make it parallel if you want):
predicted=zeros(N,S);
num_predict=10000;
for n=1:floor(N/num_predict)-1
    j=(n-1)*num_predict;
    predicted(j+1:j+num_predict,:) = predict(NN_model,Sequence_bin(j+1:j+num_predict,:),'MiniBatchSize',Batch_size);
    fprintf('.');
end
predicted(j+num_predict+1:end,:) = predict(NN_model,Sequence_bin(j+num_predict+1:end,:),'MiniBatchSize',Batch_size);
fprintf('\n');

Pred_test=predicted(Test_index,:);
Meas_test=Data(Test_index,:);
corr_results=mean(corr(Pred_test(1:end)',Meas_test(1:end)'));
RMSE_results=mean((Pred_test(1:end)'-Meas_test(1:end)').^2);
fprintf('Correlation = %f2.3, RMSE = %f2.3\n',corr_results,RMSE_results);


%plot the scatter plot
%Used dscatter which is from here:
% Reference:
% Paul H. C. Eilers and Jelle J. Goeman
% Enhancing scatterplots with smoothed densities
% Bioinformatics, Mar 2004; 20: 623 - 628.
%https://github.com/jries/SMAP/blob/master/shared/myfunctions/dscatter.m
figure(1)
dscatter(Meas_test(1:end)',Pred_test(1:end)'); %This function is available on Matlab exchange
title('Predicted vs. Measured Test Values');
xlabel('Log_1_0 Measured Binding');
ylabel('Log_1_0_Predicted Binding');
xlim([-2,2.2]);
ylim([-2,2.2]);
text(-1.8,1.8,['correlation = ',num2str(corr_results)])
text(-1.8,1.5,['RMSE = ',num2str(RMSE_results)])

%plot the loss function for test and train
figure(2);
Training_Loss=info.TrainingLoss;
Test_Loss=info.ValidationLoss;
index=~isnan(Test_Loss);
Training_Loss=Training_Loss(index);
Test_Loss=Test_Loss(index);
plot((1:length(Test_Loss))*30,[Test_Loss',Training_Loss'])
title('Loss function vs fit step')
xlabel('Number of Steps')
ylabel('Loss Function')
legend('Test Data','Training Data');

%Next create a PROJECTED set of data using a fabricated set of sequences
%with the same overall composition as the array sequences but with
%different sequences
%scramble the sequences column by column and remove anything like a
%space or Char(0)
[~,R]=size(Sequence);
fprintf('Creating the scrambled sequences\n');
NewSequence=char(zeros(N,R));
for i=1:R
    NewSequence(:,i)=Sequence(randperm(N),i);
end
for n=1:N
    a=NewSequence(n,:);
    a=a(a>='A');
    NewSequence(n,:)=char(0);
    NewSequence(n,1:length(a))=a;
end

%Now predicte the scrambled sequence binding values
%Predict values of the array sequences
NewSequence_bin=seq2bin(NewSequence,aminos); %converts amino acid sequences to one-hot vectors
projected=zeros(N,S);
num_predict=10000;
for n=1:floor(N/num_predict)-1
    a=(n-1)*num_predict;
    projected(a+1:a+num_predict,:) = predict(NN_model,NewSequence_bin(a+1:a+num_predict,:),'MiniBatchSize',Batch_size);
    fprintf('.');
end
projected(a+num_predict+1:end,:) = predict(NN_model,NewSequence_bin(a+num_predict+1:end,:),'MiniBatchSize',Batch_size);
fprintf('\n');


%Finally grab the final weight matrix from the neural network that we can use
%below instead of measured, pedicted or projected data in the multiclass
%classifier
WF=NN_model.Layers(8);
Final_weight_matrix=[WF.Weights,WF.Bias];%stick the bias on the ends of the weights
Final_weight_matrix=double(Final_weight_matrix'); %it comes back single precision otherwise


%% ***********************************************************************
%Part 3 Multiclass classification of measured, predicted, projected and final weight matrix vectors


%*****************************User set parameters for the classifier******
hidden_layers=1; %layers in NN multiclass classifier
hidden_nodes=300; %Nodes in NN multiclass classifier
Fraction_train=0.8; %fraction of samples used to train the classifier
num_rep=12; %number of noise files available per sigma value
num_features=15; %this is the number of features (e.g. peptides) selected by TTest
%*************************************************************************

aminos='ADEFGHKLNPQRSVWY';%these are the amino acids used on the array

num_disease=length(disease_names);

%Perform classifier runs
%Note that in the paper, we refit the model many times and averaged since
%there is variability in the fit.  So you may not get the same values we
%did with just one trained neural network from above.

for i_path=1:4 %run through measured, predicted, projected and final weight matrix vectors
    
    num_correct=zeros(num_disease,num_rep);
    num_tested=zeros(num_disease,num_rep);
    
    switch i_path
        case 1
            DataType='Measure Data';
            DataClass=Data;
        case 2
            DataType='Predicted Data';
            DataClass=predicted;
        case 3
            DataType='Projected Data';
            DataClass=projected;
        case 4
            DataType='Weight Matrix Vectors';
            DataClass=Final_weight_matrix;
    end
    
    [NClass,SClass]=size(DataClass);
    fprintf('\nPerforming classification for %s\n\n',DataType)
    
   
    parfor i_rep=1:num_rep
        
        %First create a set of disease indices that are the train samples for each
        %disease and a set that are the test values for each disease.
        full_train_index=false(1,SClass); %This records the full training set used - it has all the training samples for all disease as true
        full_train_index(randperm(SClass,ceil(SClass*Fraction_train)))=true(1); %this sets the right ones to true based on the fraction to train
        
        %Now do a TTest feature selection
        train_index=false(size(disease_index)); %this is the index for each disease that shows which samples we used for training
        test_index=true(size(disease_index)); %this is the index for each disease that shows which samples we used for testing
        feature_index=false(NClass,1); %this is the index of which peptides are going to be used as features
        for i_disease=1:num_disease
            train_index(i_disease,:)=full_train_index & disease_index(i_disease,:); %sets the right fraction of all samples to training
            test_index(i_disease,:)=~full_train_index & disease_index(i_disease,:); %sets the opposite samples to test
            train_index_other=full_train_index & ~disease_index(i_disease,:); %what we compare to for pvalues
            [~,pvalue]=ttest2(DataClass(:,train_index(i_disease,:))',DataClass(:,train_index_other)'); %ttest between that disease and all others
            sortp=sort(pvalue);
            ind3=pvalue<sortp(num_features);
            feature_index=feature_index|ind3'; %create index that captures top features for each disease
        end
        
        total_num_features=sum(feature_index);
        %             fprintf('%d features selected\n',total_num_features);
        Selected_Data=DataClass(feature_index,:)';
        
        %create a file of zeros and ones which has S x 6 such that the first column
        %is 1 for HCV, the second for Dengue, etc.  So these are the target values
        %of the regression
        Target=double(disease_index');
        SampleTest=Target(~full_train_index,:);
        SampleTrain=Target(full_train_index,:);
        
        [SamplePredTest,SamplePredTrain]=multiclass_classifier(Selected_Data,Target,disease_index,hidden_layers,hidden_nodes,full_train_index);
        
        [~,I]=max(SamplePredTest,[],2);
        [~,J]=max(SampleTest,[],2);
        
        for i_disease=1:num_disease
            ind1=J==I;
            num_tested(i_disease,i_rep)=sum(J==i_disease);
            num_correct(i_disease,i_rep)=sum(J(ind1)==i_disease);
        end
        
    end

    total_correct=sum((num_correct),2);
    total_tested=sum((num_tested),2);
    percent_correct=total_correct./total_tested;
    percent_err=std(num_correct./num_tested,[],2)./sqrt(total_tested);
    
    
    
    Accuracy=array2table(percent_correct');
    Error=array2table(percent_err');
    Accuracy.Properties.VariableNames = ["HCV","DEN","WNV","HBV","CHG","ND"];
    Error.Properties.VariableNames = ["HCV","DEN","WNV","HBV","CHG","ND"];
    fprintf('Final Mean Accuracies for %s\n',DataType)
    disp(Accuracy)
    fprintf('Final Errors of the Mean for %s\n',DataType)
    disp(Error)
    fprintf('___________________________________________________________\n');
end

%% Functions called*******************************************************

%**************************************************************************
function [YPredTest,YPredTrain]=multiclass_classifier(Data,Target,disease_index,hidden_layers,hidden_nodes,train_index)
%This function considers a data set with samples from multiple diseases and
%does a NN based multidisease classification

%Data is the dataset, rows are peptide, columns are samples

% Target is a one hot representation of the cohort the sample is in (e.g.,
% HCV would be [1,0,0,0,0,0]

%disease_index is a binary index with as many rows as there are diseases
%and as many columns as there are samples which says which samples are part
%of which disease

%hidden_layers is # of hidden layers in the NN

%hidden_nodes is the # of hidden nodes in the NN

% train_index is a logical index specifying which samples to use in the training set
[~,numFeatures]=size(Data);
[num_disease,~]=size(disease_index);

X=Data;
Y=Target;

%run a NN model to train using the low CV files based on ttest features
%     fprintf('Starting Neural Network multiclass classifier\n');

%set up the network architecture
layers=featureInputLayer(numFeatures,'Normalization', 'zscore'); %set up the input layer
for i_layers=1:hidden_layers %add the right number of hidden layers and activation functions
    layers=[
        layers
        fullyConnectedLayer(hidden_nodes)
        reluLayer];
end
layers=[layers
    fullyConnectedLayer(num_disease)
    regressionLayer]; %add the regression layer to the end

%define the network options
miniBatchSize = 15;
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'MaxEpochs',300, ...
    'InitialLearnRate',0.003,...
    'LearnRateDropFactor',0.9,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',2,...
    'ExecutionEnvironment','cpu',...
    'Verbose',false);

%additional options you could use:
%     'Plots','training-progress', ...     add this to see a plot during
%           the run
%      'ExecutionEnvironment','cpu',...    there are other GPU options

YTrain=Y(train_index,:);
XTrain=X(train_index,:);
XTest=X(~train_index,:);

net = trainNetwork(XTrain,YTrain,layers,options);

%Use the model to predict the values for the test samples
YPredTrain = predict(net,XTrain,'MiniBatchSize',miniBatchSize);
YPredTest = predict(net,XTest,'MiniBatchSize',miniBatchSize);
end



%*************************************************************************
%Select Random Peptides for Train and Test
function [Nindex1,Nindex2]=select_rand_peptides(Ntot,N)
%This will split the peptides into a training set (Nindex1) and a test set
%(Nindex2)
temp=randsample(Ntot,N);%Randomly picks N positions from Ntot positions
Nindex1=false(1,Ntot);
Nindex1(temp)=true(1); %create a binary index
Nindex2=~Nindex1; %create the opposite index
end

%**************************************************************************
%Convert the Sequences to a one-hot vector
function Binseq=seq2bin(Sequence,varargin)
%this function converts a sequence to a binary form.  You can include a
%list of amino acids if you want.  This is important if there is a specific
%requirement from a previously created network but the sequence you are
%converting does not have all the amino acids
N=length(Sequence);

%Is there a common linker at the C-term?  We sometimes use a GSG linker at
%the C-terminus.  If so, remove it.
k=0;
if N>=10
    for i=1:10
        if iscell(Sequence)
            a=Sequence{i};
        else
            a=Sequence(i,:);
        end
        if length(a)>2
            if strcmp(a(end-2:end),'GSG')
                k=k+1;
            end
        end
    end
end
if k==10
    linker=3;
else
    linker=0;
end

if iscell(Sequence)
    %Need to create a character array of the sequence
    Seq=char(zeros(N,20));%a character array with up to 20 characters per line
    Plen=zeros(N,1);%this is where we put the lengths of the peptide sequences
    for i=1:N
        a=Sequence{i};%convert each sequence to a string of characters
        Plen(i)=length(a)-linker;
        Seq(i,1:Plen(i))=a(1:end-linker);
    end
    R=max(Plen);
else
    Seq=Sequence;
    R=size(Sequence,2);
end

Seq=Seq(:,1:R); %get rid of extra spaces past the longest sequence
aa=Seq(1:end);
aa=aa(int32(aa)>32);%get rid of blanks

if isempty(varargin)
    aminos=unique(aa);%find the unique characters
else
    aminos=varargin{1};
end
m=length(aminos);
numseq=zeros(N,R);

%convert all the amino acids to numerical representation
for i=1:m
    numseq(Seq==aminos(i))=i;
end

v=ones(1,m);
U=diag(v); %This is a matrix with ones on the diagonal and zeros elsewehre

Binseq=zeros(N,R*m);
for n=1:N
    pep2D=zeros(m,R);
    for r=1:R
        if numseq(n,r)>0
            pep2D(:,r)=U(numseq(n,r),:);
        end
    end
    Binseq(n,:)=pep2D(1:end);
end
end


