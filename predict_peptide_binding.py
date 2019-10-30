# ---Neural Network for Peptide Binding Prediction--- #
#
# Architecture:
#    Input: Sequence represented as [residue number x amino acid] matrix
#    Model:
#        [1] Linear encoder converts one-hot amino acid representation to dense representation
#        [2] Feed-forward neural network with two hidden layers
#        [3] Output regression layer predicts binding value


# ~~~~~~~~~~MODULES~~~~~~~~~~ #
import datetime
import numpy as np
import os
import pandas as pd
import random
import re
import torch
from torch.multiprocessing import Pool
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

# Store current module and parameter scope
moduleScope = dir()


# ~~~~~~~~~~PARAMETERS~~~~~~~~~~ #
aminoEncoder = 10  # number of features to describe amino acids (default: 10)
chemParams = False  # use chem.txt as amino acid representation (default: False)
filename = 'data/FNR.csv'  # path to sequence and binding data (default: 'data/FNR.csv')
hiddenLayers = 2  # number of hidden layers in neural network (default: 2)
hiddenNodes = 100  # number of nodes per hidden layer of neural network (default: 100)
multipleRuns = 1  # repeat training multiple times in parallel (default: 1)
testingMode = False  # path to pretrained 'Model.pth' neural network (default: False)
trainFraction = 0.9  # fraction of non-saturated data for training (default: 0.9)
trainSteps = 50000  # number of training steps (default: 50000)
weightFolder = 'fits'  # directory name to save weights and biases (default: 'fits')
weightSave = False  # save weights to file (default: False)

# If chemParams is True, assert that aminoEncoder equals number of chemical parameters
if chemParams:
    with open('data/chem.txt', 'r') as f:
        f.readline()
        count = len(f.readline().strip().split('\t'))
        assert aminoEncoder == count, f'Set aminoEncoder to number {count}!'
    del count, f

# Store parameter settings
paramScope = [x for x in dir() if x not in moduleScope + ['moduleScope']]
paramLocals = locals()
paramDict = {x: paramLocals[x] for x in paramScope}


# ~~~~~~~~~~PARALLELIZATION~~~~~~~~~~ #
def training(sample):

    # Randomize pytorch and numpy seed states
    torch.manual_seed(random.randint(1, 10**6))
    np.random.seed(random.randint(1, 10**6))

    # Import matplotlib into new environment
    import matplotlib.pyplot as plt

    # Define amino acid letter codes
    aminoAcids = 'ADEFGHKLNPQRSVWY'

    # Represent amino acids with measured chemical parameters
    if chemParams:
        chem_params = pd.read_csv('data/chem.txt', delimiter='\t', header=0, index_col=0, skiprows=1)
        chem_params = torch.from_numpy(chem_params.loc[list(aminoAcids)].values).float()
    else:
        chem_params = 0

    # Import and clean sequence data where first column is the sequences followed by the binding data
    data = pd.read_csv(filename, header=None)
    data[0].replace(re.compile('[^' + aminoAcids + ']'), '', inplace=True)

    # Remove trailing GSG from sequences
    if sum(data[0].str[-3:] == 'GSG') / len(data) > 0.9:
        data[0] = data[0].str[:-3]    

    # Remove sequences shorter than 3 residues in length
    data.drop(data[0].index[data[0].str.len() < 3].tolist(), inplace=True)

    # Assign binary vector to each amino acid
    amino_dict = {n: m for (m, n) in enumerate(aminoAcids)}

    # Create binary sequence matrix representation
    max_len = int(data[0].str.len().max())
    sequences = np.zeros((len(data), len(aminoAcids) * max_len), dtype='int8')
    for (n, m) in enumerate(data[0]):
        amino_ind = [amino_dict[j] + (i * len(aminoAcids)) for (i, j) in enumerate(m)]
        sequences[n][amino_ind] = 1

    # Add 100 and take base-10 logarithm of binding data
    data = np.log10(data[1].values + 100)

    # Combine sequence and data
    train_test = np.concatenate((sequences, np.transpose([data])), axis=1)
    train_test = train_test[train_test[:, -1].argsort()]

    # Shuffle all data excluding the top 2%
    exclude_value = 0.98 * (train_test[-1, -1] - train_test[0, -1]) + train_test[0, -1]
    exclude_ind = np.abs(train_test[:, -1] - exclude_value).argmin()
    np.random.shuffle(train_test[:exclude_ind])

    # Train and test sets
    train_xy = np.copy(train_test[:int(trainFraction * exclude_ind), :])
    train_xy = train_xy[train_xy[:, -1].argsort()]
    test_xy = np.copy(train_test[int(trainFraction * exclude_ind):, :])

    # Find bin indices for uniformly distributed batch gradient descent
    bin_data = np.linspace(train_xy[0][-1], train_xy[-1][-1], 100)
    bin_ind = [np.argmin(np.abs(x - train_xy[:, -1])) for x in bin_data]
    bin_ind = np.append(bin_ind, len(train_xy))

    # Convert to PyTorch variable tensors
    train_seq = torch.from_numpy(train_xy[:, :-1]).float()
    train_data = torch.from_numpy(train_xy[:, -1]).float()
    test_seq = torch.from_numpy(test_xy[:, :-1]).float()
    test_data = torch.from_numpy(test_xy[:, -1]).float()


# ~~~~~~~~~~NEURAL NETWORK~~~~~~~~~~ #
    class NeuralNet(nn.Module):

        def __init__(self, node, layers):
            super().__init__()

            # Layer nodes
            amino = aminoEncoder if aminoEncoder else 1
            hidden = max_len * aminoEncoder if aminoEncoder else max_len * len(aminoAcids)

            # Network layers
            if chemParams:
                self.AminoLayer = nn.Linear(amino, amino, bias=True)
            else:
                self.AminoLayer = nn.Linear(len(aminoAcids), amino, bias=False)
            self.HiddenLayers = nn.ModuleList([nn.Linear(hidden, node, bias=True)])
            self.HiddenLayers.extend([nn.Linear(node, node, bias=True) for _ in range(layers - 1)])
            self.OutputLayer = nn.Linear(node, 1, bias=True)

        def forward(self, seq):
            if aminoEncoder:
                seq = seq.view(-1, len(aminoAcids))
                if chemParams:
                    seq = torch.mm(seq, chem_params)
                seq = self.AminoLayer(seq)
                seq = seq.view(-1, max_len * aminoEncoder)
            for x in self.HiddenLayers:
                seq = functional.relu(x(seq))
            return self.OutputLayer(seq)

    net = NeuralNet(node=hiddenNodes, layers=hiddenLayers)
    print('\nARCHITECTURE:')
    print(net)

    # Loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)


# ~~~~~~~~~~TRAINING~~~~~~~~~~ #
    if testingMode:

        # Load pretrained weights
        net.load_state_dict(torch.load(testingMode))

        # Run test set through optimized neural network and determine correlation coefficient
        test_prediction = torch.squeeze(net(torch.cat((train_seq, test_seq)))).data.numpy()
        test_real = torch.cat((train_data, test_data)).data.numpy()
        correlation = np.corrcoef(test_real, test_prediction)[0, 1]
        print('Correlation Coefficient: %.3f' % correlation)

    else:

        # Record best network weights and loss
        best_net = net
        best_loss = np.inf
        print('\nTRAINING:')
        for i in range(trainSteps + 1):

            # Select indices for training
            train_ind = [np.random.randint(bin_ind[i], bin_ind[i + 1] + 1)
                            for i in range(len(bin_ind) - 1)]
            train_ind[-1] = train_ind[-1] - 1

            # Calculate loss
            if i < (trainSteps - 10):
                train_out = net(train_seq[train_ind])
                loss = loss_function(torch.squeeze(train_out), train_data[train_ind])

            else:
                train_out = net(train_seq)
                loss = loss_function(torch.squeeze(train_out), train_data)

                # Remember the best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_net = net

            # Weight optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Report training progress every 100 steps
            if i % 100 == 0:

                # Restore best model on final loop
                if i == (trainSteps + 1):
                    net = best_net

                # Select 1000 training and test indices for progress report
                train_batch = random.sample(range(train_seq.shape[0]), 1000)
                test_batch = random.sample(range(test_seq.shape[0]), 1000)

                # Train and test binding predictions
                train_prediction = torch.squeeze(net(train_seq[train_batch]))
                test_prediction = torch.squeeze(net(test_seq[test_batch]))

                # Train and test accuracies
                train_accuracy = torch.abs(train_prediction - train_data[train_batch])
                train_accuracy = len(torch.nonzero(train_accuracy < 0.2)) / len(train_accuracy)
                test_accuracy = torch.abs(test_prediction - test_data[test_batch])
                test_accuracy = len(torch.nonzero(test_accuracy < 0.2)) / len(test_accuracy)

                # Report train and test accuracies
                print('Step %5d: train|test accuracy: %.2f|%.2f' %
                      (i, train_accuracy, test_accuracy))

        # Run test set through optimized neural network and determine correlation coefficient
        test_prediction = torch.squeeze(net(test_seq)).data.numpy()
        test_real = test_data.data.numpy()
        correlation = np.corrcoef(test_real, test_prediction)[0, 1]
        print('Correlation Coefficient: %.3f' % correlation)


# ~~~~~~~~~~PLOTTING~~~~~~~~~~ #
    # Extract weights from model
    amino_layer = net.AminoLayer.weight.data.transpose(0, 1).numpy()
    hidden_layer = [[x.weight.data.transpose(0, 1).numpy(),
                     x.bias.data.numpy()]for x in net.HiddenLayers]
    output_layer = [net.OutputLayer.weight.data.transpose(0, 1).numpy(),
                    net.OutputLayer.bias.data.numpy()]

    # Turn off interactive mode
    plt.ioff()

    # Scatter plot of predicted vs real
    fig1 = plt.figure()
    plt.scatter(test_real, test_prediction, s=1, color='b')
    plt.plot([min(test_real), max(test_real)],
             [min(test_real), max(test_real)], color='k')
    plt.xlabel('Real', fontsize=15)
    plt.ylabel('Prediction', fontsize=15)
    plt.title('Correlation Coefficient: %.3f' % correlation, fontsize=15)

    # Amino acid similarity matrix
    if not chemParams:
        amino_similar = np.linalg.norm(amino_layer, axis=1)
        amino_similar = np.array([aminoEncoder * [magnitude] for magnitude in amino_similar])
        amino_similar = np.dot((amino_layer / amino_similar),
                                np.transpose(amino_layer / amino_similar))
        fig2 = plt.matshow(amino_similar, cmap='coolwarm')
        plt.xticks(range(len(aminoAcids)), aminoAcids)
        plt.yticks(range(len(aminoAcids)), aminoAcids)
        plt.colorbar()
        plt.clim(-1, 1)


# ~~~~~~~~~~SAVE MODEL~~~~~~~~~~ #
    if weightSave:

        # Create path to date folder
        date = sorted([x for x in os.listdir(weightFolder) if ''.join(x.split('-')).isdigit()])[-1]
        date_folder = weightFolder + '/' + date

        # Create path to run folder
        old_runs = sum(('Run' in x for x in os.listdir(date_folder)))
        run_folder = weightFolder + '/' + date + '/Run' + str(old_runs) + '-' + filename.split('/')[-1][:-4]

        # Create path to new sample folder
        directory = run_folder + '/Sample' + str(abs(sample))
        os.makedirs(directory)

        # Save weights and biases to csv files
        np.savetxt(directory + '/W1.txt', amino_layer, delimiter=',')
        if chemParams:
            np.savetxt(directory + '/B1.txt',
                       net.AminoLayer.bias.data.numpy(), delimiter=',')
        for (m, n) in enumerate(hidden_layer):
            np.savetxt(directory + '/W' + str(m + 2) + '.txt', n[0], delimiter=',')
            np.savetxt(directory + '/B' + str(m + 2) + '.txt', n[1], delimiter=',')
        np.savetxt(directory + '/WF.txt', output_layer[0], delimiter=',')
        np.savetxt(directory + '/BF.txt', output_layer[1], delimiter=',')

        # Save correlation coefficient to file
        with open(directory + '/CORR.txt', 'w') as f:
            f.write(str(correlation))

        # Save parameter settings
        with open(run_folder + '/Parameters.txt', 'w') as f:
            f.write('#~~~ARCHITECTURE~~~#\n')
            f.write(str(net))
            f.write('\n\n#~~~PARAMETERS~~~#\n')
            for m, n in paramDict.items():
                f.write(str(m) + ': ' + str(n) + '\n')

        # Save figures
        fig1.savefig(directory + '/Correlation.png', bbox_inches='tight')
        if not chemParams:
            fig2.figure.savefig(directory + '/Similarity.png', bbox_inches='tight')

        # Save model
        torch.save(net.state_dict(), directory + '/Model.pth')

    # Show figures
    else:
        plt.show()


if __name__ == '__main__':

    # Generate file structure
    currentDate = datetime.datetime.today().strftime('%Y-%m-%d')
    if weightSave:

        # Create parent folder
        if not os.path.exists(weightFolder):
            os.makedirs(weightFolder)

        # Create date folder
        dateFolder = weightFolder + '/' + currentDate
        if not os.path.exists(dateFolder):
            os.makedirs(dateFolder)

        # Create run folder
        oldRuns = sum(('Run' in x for x in os.listdir(dateFolder)))
        runFolder = dateFolder + '/Run' + str(oldRuns + 1) + '-' + filename.split('/')[-1][:-4]
        os.makedirs(runFolder)

    # Single training run or multiple training runs in parallel
    if multipleRuns == 1:
        training(1)
    else:
        pool = Pool()
        pool.map(training, range(1, multipleRuns + 1))
