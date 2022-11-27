#=== MODULES ===#
import argparse
import datetime
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim


#=== MATLAB INTERFACE ===#
# Import parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('params', nargs='?')
args = parser.parse_args()

# Create dictionary of imported parameters
parameter_imports = {}
if args.params:
    argument_parameters = pd.read_csv(args.params, header=None, index_col=0, squeeze=True).str.strip()
    for index, value in argument_parameters.iteritems():
        if value.isdigit():
            param = int(value)
        elif value[0].isdigit():
            param = float(value)
        elif value.lower() == 'true':
            param = True
        elif value.lower() == 'false':
            param = False
        else:
            param = value
        parameter_imports[index] = param

# Store current module and parameter scope
module_scope = dir()

# Replace parameters with imported settings
if parameter_imports:
    globals().update(parameter_imports)

# Store parameter settings
parameter_scope = [x for x in dir() if x not in module_scope + ['module_scope']]
parameter_locals = locals()
param_dictionary = {x: parameter_locals[x] for x in parameter_scope}


#=== FEED-FORWARD NEURAL NETWORK ===#
class NeuralNetwork2020():
    """Neural Network for Peptide Binding Prediction - Taguchi & Woodbury et al, Combinatorial Science (2020)

    Architecture:
        Input: Sequence represented as [residue number x amino acid] matrix
        Model:
            [1] Linear encoder converts one-hot amino acid representation to dense representation
            [2] Feed-forward neural network with multiple hidden layers
            [3] Output regression layer predicts binding value
    """

    def __init__(self, chem_encoder=False, data='data/FNR.csv', encoder_nodes=10,
                 evaluation_mode=False, hidden_layers=2, hidden_nodes=100, save_weights=False,
                 train_fraction=0.9, train_steps=50000, train_test_split=[], weight_folder='fits'):
        """Parameter and file structure initialization
        
        Keyword Arguments:
            chem_encoder {str} -- path to amino acid properties to use as encoder (default: {False})
            data {str/df} -- file path or dataframe of sequences and data (default: {'data/FNR.csv'})
            encoder_nodes {int} -- number of features to describe amino acids (default: {10})
            evaluation_mode {str} -- path to pretrained 'Model.pth' neural network (default: {False})
            hidden_layers {int} -- number of hidden layers in neural network (default: {2})
            hidden_nodes {int} -- number of nodes per hidden layer of neural network (default: {100})
            save_weights {bool} -- save weights to file (default: {False})
            train_fraction {float} -- fraction of non-saturated data for training (default: {0.9})
            train_steps {int} -- number of training steps (default: {50000})
            train_test_split {list} -- train (0) and test (1) split assignments (default: {[]})
            weight_folder {str} -- directory name to save weights and biases (default: {'fits'})
        """
        # Initialize variables
        self.chem_encoder = chem_encoder
        self.data = data
        self.encoder_nodes = encoder_nodes
        self.evaluation_mode = evaluation_mode
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.save_weights = save_weights
        self.train_fraction = train_fraction
        self.train_steps = train_steps
        self.train_test_split = train_test_split
        self.weight_folder = weight_folder

        # Assert that encoder_nodes is set correctly
        if self.chem_encoder:
            with open('data/chem.txt', 'r') as f:
                properties = f.readlines()[-1].split()
                properties = len([x for x in properties if x.lstrip('-').replace('.', '', 1).isdigit()])
                assert self.encoder_nodes == properties, f'Set encoder_nodes={properties} to match chem_encoder!'
            del properties, f
        
        # Store parameter settings
        self.settings = {key: value for key, value in locals().items() if key != 'self'}

        # Import train and test split assignments
        if len(self.train_test_split):
            self.train_test_split = np.array(self.train_test_split, dtype=int)
            assert set(self.train_test_split) == {0, 1}, 'Only 0 and 1 allowed in train_test_split!'

        # Generate file structure
        current_date = datetime.datetime.today().strftime('%Y-%m-%d')
        if self.save_weights:

            # Create parent folder
            if not os.path.exists(self.weight_folder):
                os.makedirs(self.weight_folder)

            # Create date folder
            date_folder = self.weight_folder + '/' + current_date
            if not os.path.exists(date_folder):
                os.makedirs(date_folder)

            # Create run folder
            old_runs = sum(('Run' in x for x in os.listdir(date_folder)))
            filename = data.split('/')[-1][:-4] if isinstance(data, str) else data.name
            self.run_folder = date_folder + '/Run' + str(old_runs + 1) + '-' + filename
            os.makedirs(self.run_folder)

    def fit(self, sample=1):
        """Train or evaluate neural network
        
        Keyword Arguments:
            sample {int} -- sample identification number (default: {1})
        """

        # Randomize pytorch and numpy seed states
        torch.manual_seed(random.randint(1, 10**6))
        np.random.seed(random.randint(1, 10**6))

        # Import matplotlib into new environment
        import matplotlib.pyplot as plt

        # Define amino acid letter codes
        amino_acids = 'ADEFGHKLNPQRSVWY'

        # Represent amino acids with measured chemical parameters
        if self.chem_encoder:
            chem_params = pd.read_csv('data/chem.txt', delimiter='\t', header=0, index_col=0, skiprows=1)
            chem_params = torch.from_numpy(chem_params.loc[list(amino_acids)].values).float()
        else:
            chem_params = 0

        # Import and clean sequence data where first column is the sequences followed by the binding data
        data = pd.read_csv(self.data, header=None) if isinstance(self.data, str) else self.data
        data[0].replace(re.compile(f'[^{amino_acids}]'), '', inplace=True)

        # Check that there are no identical sequences
        assert len(data) == len(data.groupby(0).mean()), 'Duplicate sequences found!'

        # Remove trailing GSG from sequences
        if sum(data[0].str[-3:] == 'GSG') / len(data) > 0.9:
            data[0] = data[0].str[:-3]

        # Assign binary vector to each amino acid
        amino_dict = {n: m for (m, n) in enumerate(amino_acids)}

        # Create binary sequence matrix representation
        max_len = int(data[0].str.len().max())
        sequences = np.zeros((len(data), len(amino_acids) * max_len), dtype='int8')
        for (n, m) in enumerate(data[0]):
            amino_ind = [amino_dict[j] + (i * len(amino_acids)) for (i, j) in enumerate(m)]
            sequences[n][amino_ind] = 1

        # Add 100 and take base-10 logarithm of binding data
        data = np.log10(data[1].values + 100)

        # Combine sequence and data
        train_test = np.concatenate((sequences, np.transpose([data])), axis=1)

        # Randomly generate split between train and test sets if not manually specified
        if not len(self.train_test_split):

            # Exclude saturated binding values from train set
            saturation_threshold = 0.98 * np.ptp(train_test[:, -1]) + min(train_test[:, -1])

            # Assign train and test set indices
            nonsaturated = np.where(train_test[:, -1] <= saturation_threshold)[0]
            train_size = int(self.train_fraction * len(nonsaturated))
            train_split = np.random.choice(nonsaturated, train_size, replace=False)
            self.train_test_split = np.ones(len(data), dtype=int)
            self.train_test_split[train_split] = 0

        # Split into train and test sets
        train_xy = np.copy(train_test[[x == 0 for x in self.train_test_split]])
        test_xy = np.copy(train_test[[x == 1 for x in self.train_test_split]])

        # Find bin indices for uniformly distributed batch gradient descent
        train_xy = train_xy[train_xy[:, -1].argsort()]
        bin_data = np.linspace(train_xy[0][-1], train_xy[-1][-1], 100)
        bin_ind = [np.argmin(np.abs(x - train_xy[:, -1])) for x in bin_data]
        bin_ind = np.append(bin_ind, len(train_xy))

        # Convert to PyTorch variable tensors
        train_seq = torch.from_numpy(train_xy[:, :-1]).float()
        train_data = torch.from_numpy(train_xy[:, -1]).float()
        test_seq = torch.from_numpy(test_xy[:, :-1]).float()
        test_data = torch.from_numpy(test_xy[:, -1]).float()


        # Neural network architecture
        class Architecture(nn.Module):

            def __init__(self, encoder, nodes, layers):
                super().__init__()

                # Layer nodes
                self.encoder = encoder
                self.hidden = max_len * encoder if encoder else max_len * len(amino_acids)

                # Amino acid encoder
                if encoder and chem_params:
                    self.encoder_layer = nn.Linear(encoder, encoder, bias=True)
                elif encoder:
                    self.encoder_layer = nn.Linear(len(amino_acids), encoder, bias=False)
                
                # Hidden layers
                self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden, nodes, bias=True)])
                self.hidden_layers.extend([nn.Linear(nodes, nodes, bias=True) for _ in range(layers - 1)])

                # Output layer
                self.output_layer = nn.Linear(nodes, 1, bias=True)

            def forward(self, seq):
                if self.encoder:
                    seq = seq.view(-1, len(amino_acids))
                    if chem_params:
                        seq = torch.mm(seq, chem_params)
                    seq = self.encoder_layer(seq)
                    seq = seq.view(-1, max_len * self.encoder)
                for x in self.hidden_layers:
                    seq = functional.relu(x(seq))
                return self.output_layer(seq)

        net = Architecture(encoder=self.encoder_nodes, nodes=self.hidden_nodes, layers=self.hidden_layers)
        print('\nARCHITECTURE:')
        print(net)

        # Loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)

        # Training
        if self.evaluation_mode:

            # Load pretrained weights
            net.load_state_dict(torch.load(self.evaluation_mode))

            # Run test set through optimized neural network and determine correlation coefficient
            train_real = train_data.data.numpy()
            test_real = test_data.data.numpy()
            train_prediction = torch.squeeze(net(train_seq)).data.numpy()
            test_prediction = torch.squeeze(net(test_seq)).data.numpy()
            correlation = np.corrcoef(test_real, test_prediction)[0, 1]
            print('Correlation Coefficient: %.3f' % correlation)

            return (train_real, train_prediction, test_real, test_prediction)

        else:

            # Record best network weights and loss
            best_net = net
            best_loss = np.inf
            losses = []
            print('\nTRAINING:')
            for i in range(self.train_steps + 1):

                # Select indices for training
                train_ind = [np.random.randint(bin_ind[i], bin_ind[i + 1] + 1)
                             for i in range(len(bin_ind) - 1)]
                train_ind[-1] = train_ind[-1] - 1

                # Calculate loss
                if i < (self.train_steps - 10):
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
                    if i == (self.train_steps + 1):
                        net = best_net

                    # Select 1000 training and test indices for progress report
                    train_batch = random.sample(range(train_seq.shape[0]), 1000)
                    test_batch = random.sample(range(test_seq.shape[0]), 1000)

                    # Train and test binding predictions
                    train_prediction = torch.squeeze(net(train_seq[train_batch]))
                    test_prediction = torch.squeeze(net(test_seq[test_batch]))

                    # Record train and test losses
                    train_loss = loss_function(train_prediction, train_data[train_batch])
                    test_loss = loss_function(test_prediction, test_data[test_batch])
                    losses.append((i, train_loss, test_loss))

                    # Train and test accuracies
                    train_accuracy = torch.abs(train_prediction - train_data[train_batch])
                    train_accuracy = len(torch.nonzero(train_accuracy < 0.2)) / len(train_accuracy)
                    test_accuracy = torch.abs(test_prediction - test_data[test_batch])
                    test_accuracy = len(torch.nonzero(test_accuracy < 0.2)) / len(test_accuracy)

                    # Report train and test accuracies
                    print(f'Step {i:5d}: train|test accuracy - {train_accuracy:.2f}|{test_accuracy:.2f}')

            # Run test set through optimized neural network and determine correlation coefficient
            test_prediction = torch.squeeze(net(test_seq)).data.numpy()
            test_real = test_data.data.numpy()
            correlation = np.corrcoef(test_real, test_prediction)[0, 1]
            print(f'Correlation Coefficient: {correlation:.3f}')

        # Extract weights from model
        if self.encoder_nodes:
            encoder_layer = net.encoder_layer.weight.data.transpose(0, 1).numpy()
        hidden_layer = [[x.weight.data.transpose(0, 1).numpy(),
                        x.bias.data.numpy()]for x in net.hidden_layers]
        output_layer = [net.output_layer.weight.data.transpose(0, 1).numpy(),
                        net.output_layer.bias.data.numpy()]

        # Turn off interactive mode
        plt.ioff()

        # Scatter plot of predicted vs real
        fig1 = plt.figure()
        plt.scatter(test_real, test_prediction, s=1, color='b')
        plt.plot([min(test_real), max(test_real)],
                 [min(test_real), max(test_real)], color='k')
        plt.xlabel('Real', fontsize=15)
        plt.ylabel('Prediction', fontsize=15)
        plt.title(f'Correlation Coefficient: {correlation:.3f}', fontsize=15)

        # Amino acid similarity matrix
        if self.encoder_nodes and not self.chem_encoder:
            amino_similar = np.linalg.norm(encoder_layer, axis=1)
            amino_similar = np.array([self.encoder_nodes * [magnitude] for magnitude in amino_similar])
            amino_similar = np.dot((encoder_layer / amino_similar),
                                    np.transpose(encoder_layer / amino_similar))
            fig2 = plt.matshow(amino_similar, cmap='coolwarm')
            plt.xticks(range(len(amino_acids)), amino_acids)
            plt.yticks(range(len(amino_acids)), amino_acids)
            plt.colorbar()
            plt.clim(-1, 1)

        # Save run to file
        if self.save_weights:

            # Create path to new sample folder
            directory = f'{self.run_folder}/Sample{str(abs(sample))}'
            os.makedirs(directory)

            # Save train test split
            with open(f'{directory}/TrainTestSplit.txt', 'w') as f:
                f.writelines(f'{x}\n' for x in self.train_test_split)

            # Save weights and biases to csv files
            if self.encoder_nodes:
                np.savetxt(f'{directory}/W1.txt', encoder_layer, delimiter=',')
            if self.chem_encoder:
                np.savetxt(f'{directory}/B1.txt', net.AminoLayer.bias.data.numpy(), delimiter=',')
            for (m, n) in enumerate(hidden_layer):
                np.savetxt(f'{directory}/W{str(m + 2)}.txt', n[0], delimiter=',')
                np.savetxt(f'{directory}/B{str(m + 2)}.txt', n[1], delimiter=',')
            np.savetxt(f'{directory}/WF.txt', output_layer[0], delimiter=',')
            np.savetxt(f'{directory}/BF.txt', output_layer[1], delimiter=',')

            # Save correlation coefficient to file
            with open(f'{directory}/Correlation.txt', 'w') as f:
                f.write(str(correlation))
            
            # Save training and testing losses
            with open(f'{directory}/Loss.txt', 'w') as f:
                for loss in losses:
                    f.write(f'Step {loss[0]:5d}: train|test loss - {loss[1]:.5f}|{loss[2]:.5f}\n')

            # Save parameter settings
            with open(f'{self.run_folder}/Parameters.txt', 'w') as f:
                f.write('#~~~ARCHITECTURE~~~#\n')
                f.write(str(net))
                f.write('\n\n#~~~PARAMETERS~~~#\n')
                for m, n in self.settings.items():
                    f.write(f'{str(m)}: {str(n)}\n')

            # Save figures
            fig1.savefig(f'{directory}/Correlation.png', bbox_inches='tight')
            plt.close(fig1)
            if self.encoder_nodes and not self.chem_encoder:
                fig2.figure.savefig(f'{directory}/Similarity.png', bbox_inches='tight')
                plt.close()

            # Save model
            torch.save(net.state_dict(), f'{directory}/Model.pth')

        # Show figures
        else:
            plt.show()


#=== RUN PEPTIDE-ARRAY-ML ===#
if __name__ == '__main__':
    samples = pd.read_csv(param_dictionary['data'], header=None).values.shape[1]
    neural_network = NeuralNetwork(**param_dictionary)
    pool = Pool()
    pool.map(neural_network.fit, range(1, samples + 1 if param_dictionary['fit_sample'] else 2))
