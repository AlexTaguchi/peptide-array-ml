function NN_fit7(Data,Sequence,Train,fit_sample,hidden_layers,hidden_nodes,log_shift,train_steps,weight_folder,Data_write)
%This function runs an NN fit with all of the columns in Data and assuming
%that the Sequences match the rows.  Most of the inputs to the python
%program are hardwired below except the set in the function call

%note this version has Data_write which is a flag that determines whether
%it writes the data file out or uses the exsting one.  Often, we are doing
%multiple runs without changing the data file, only which sequences are
%used as the train vs. test

%***in this version, it does not print out or return the predicted values

%in the old version I included the training fraction but that is overridden
%by the Train set so I removed it in this version

%These are the values specified in the parameter file in the formate that
%the file must be written
% sequences,data/DM1A_sequence.csv
% context,data/DM1A_concentration.csv
% data,data/DM1A_data.csv
% amino_acids,ADEFGHKLNPQRSVWY
% amino_embedder_nodes,10
% batch_size,100
% chemical_embedder,false
% evaluate_model,false
% fit_sample,false
% hidden_layers,2
% hidden_nodes,100
% layer_freeze,0
% learn_rate,0.001
% log_shift,100
% saturation_threshold,0.99
% save_predictions, false
% save_weights,true
% sequence_embedder_nodes,50
% train_fraction,0.9
% train_steps,50000
% transfer_learning,false
% weight_folder,fits


%
%Notes

%sequences: One has two columns, the sequence of the peptide and a zero or
%one specifying train(0) or test (1).  If you do not specify the binary
%designation of test and train, it will randomly pick based on the train
%fraction (see below)
%
%context: The next is a context file which has a vector of numbers for
%each sequence that relates to context such as concentration, target
%(typically a one-hot representation), target properties, maybe position of
%the peptide in the array -- whatever you want to tell the algorithm that
%relates to that sequence and the data you will provide for that sequence.
%
%data:  This is the data associated with the measurement.  It can be
%multiple columns if there are multiple measurements for that sequence.

%amino_acids:  This is a string containing the characters that represent
%the amino acids used

%batch_size:  This is the number of sequence/data pairs used in each
%iteration

%chem_encoder:  If you want to put in a permanent encoder file, usually
%representing specific propoerties of the amino acids, you can use this
%option and replace false with that file name.

%encoder_nodes: The number of values used to represent each amino acid.
%The encoder is a matrix that is the number of nodes x the number of
%different amino acids used.  It represents an information bottleneck and a
%means of extracting information about the way the NN represents the amino
%acids.  Unless you specify it with chem_encoder, the program creates it
%and it is part of the optimization.

%evaluation_mode:false to fit, put in a path to a Model.pth file to have it
%evaluate for specific sequences

%hidden_layers:  number of hidden layers in the neural network (does not
%include the encoder and the output layer)

%hidden_nodes: width of the hidden layers (all the same)

%layer_freeze: when you do transfer learning, you can specify how many
%layers are frozen.  1 would mean the encoder was frozen. 2 would mean the
%encoder and the first hidden layer were frozen.  Etc all the way out to
%the last hidden layer.

%learn_rate:  the rate at which the optimization proceeds.  A faster rate
%means it takes bigger jumps.  Generally the number given is a starting
%point.  The algorithm modifies the rate as the optimization procedes and
%gets smaller when near optimium

%train_fraction: if you do not specify the train and test sequences in
%"sequences", this specifies what fraction to train with, chosen randomly

%train_steps: this is the number of optimization steps used

%tranfer learning: false means no transfer learning, or insert a file name.
%If you want to use the most recent fit for transfer learning you
%can find the path in fits.log.  The model itself is contained in Model.pth
%and you replace "false" in the transfer learning with Model.pth and its
%path: fits/2020-11-25/Run6-DM1A_sequence/Sample1/Model.pth

%weight_folder: the name of the folder it will store the weights in

%weight_save: true if you want it to save the weights in the folder

main_dir=cd;
[N,S]=size(Data);

%These are the default parameters that will appear in the user input table
sequence_file='sequence_file.csv'; %sequence file containing sequences and test/train instructions
context_file='false'; %context file containing info about the sequence and the data
data_file='data_file.csv'; %the file containing the data for training
amino_acids='ADEFGHKLNPQRSVWY'; %list of amino acids used
amino_embedder_nodes=10;%number of features to describe amino acids - default: 10
batch_size=100; %the number of sequence/data pairs used each iteration
chemical_embedder='false'; %whether you import an encoder file or let the program make it
evaluation_model='false';%if a model file provided than it evalutes the input sequences based that
%fit_sample='false'; %if true then it fits all the samples individually, else it fits them as one loss function
% hidden_layers=5;% number of hidden layers
% hidden_nodes=250;% width of hidden layers
layer_freeze=0; %number of layers to freeze in transfer learning
learn_rate=0.001; %size of initial optimization steps
% log_shift=0.01; %what it addes to the data before taking the log
saturation_threshold=0.99;%anything above this is put in test set if you don't provide the test set definition
save_predictions='false';%save the predictions...this is a really big file
save_weights='true';%save weights to file - default: false
sequence_embedder_nodes=hidden_nodes; %this lets you set the final hidden layer width individually
train_fraction=.99;%fraction of the sequences to use in training
% train_steps=50000;% number of training steps - default: 20000
transfer_learning='false'; %whether to use an existing model as a starting point
% weight_folder='fits';% name of folder for saving weights and biases


%define the arguement file name
arg_file='args.txt';

%There are three files used by the python program.
%The first is a sequence file that has two columns, one
%is the sequence and one is the training code (0 or 1), the next is the
%context file with a series of comma separated values for context and the
%last is the data file.

fprintf('Building the arrays that are needed to create the files for the machine learning\n')

if Data_write
    %write the data file
    fprintf('writing the data file\n');
    dlmwrite(data_file,Data);
end

%write the sequence/training file
fprintf('writing the sequence/training file');
[numdata,~]=size(Data);
fid1=fopen(sequence_file,'w');
j=1;
for i=1:numdata
    fprintf(fid1,'%s,%d\n',Sequence{i},Train(i));
    j=j+1;
    if j>1000000
        fprintf('.');
        j=1;
    end
end
fprintf('\n');
fclose all;

%************************************************

%Set up the initial values of the parameter file that is fed into pytorch
args = {...
    'sequences',sequence_file;...
    'context',context_file;...
    'data',data_file;...
    'amino_acids',amino_acids;...
    'amino_embedder_nodes',amino_embedder_nodes;...
    'batch_size',batch_size;...
    'chemical_embedder',chemical_embedder;...
    'evaluate_model',evaluation_model;...
    'fit_sample',fit_sample;...
    'hidden_layers',hidden_layers;...
    'hidden_nodes',hidden_nodes;...
    'layer_freeze',layer_freeze;...
    'learn_rate',learn_rate;...
    'log_shift',log_shift;...
    'saturation_threshold',saturation_threshold;...
    'save_predictions',save_predictions;...
    'save_weights',save_weights;...
    'sequence_embedder_nodes',sequence_embedder_nodes;...
    'train_fraction',train_fraction;...
    'train_steps',train_steps;...
    'transfer_learning',transfer_learning;...
    'weight_folder',weight_folder;...
    };

%write the parameter file
fprintf('writing the parameter file\n');
write_MLparms_to_file(args,arg_file)



%now run the nn fit
fprintf('Starting neural network...\n');
status=system('python peptide_array_ml\__init__.py args.txt','-echo');

% %Go grab the results of the NN fit
% fid1=fopen([main_dir,'\',weight_folder,'\',weight_folder,'.log']);
% fit_path=fgetl(fid1);%this reads the path to the fit results from above
% a=strfind(fit_path,',');
% fit_path=fit_path(a+1:end);
% 
% if strcmp(fit_sample,'false')
%     %Read in the predicted values from the fit
%     WFarray=dlmread([main_dir,'\',fit_path,'\WF.txt']);
% else
%     b=strfind(fit_path,'Sample');
%     fit_path=fit_path(1:b-1);
%     WF1=dlmread([fit_path,'Sample1\WF.txt']);
%     WFarray=zeros(length(WF1),S);
%     for s=1:S
%         new_path=[fit_path,'Sample',num2str(s)];
%         WFarray(:,s)=dlmread([main_dir,'/',new_path,'/WF.txt']);
%     end
end



   



