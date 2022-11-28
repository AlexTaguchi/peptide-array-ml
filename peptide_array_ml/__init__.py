#=== MODULES ===#
import argparse
import datetime
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import random
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
parameter_settings = {x: parameter_locals[x] for x in parameter_scope}


#=== FEED-FORWARD NEURAL NETWORK ===#
class NeuralNetwork():
    """Neural Network for Modeling Peptide Array Sequence and Binding Values

    Architecture:
        Input: Sequences represented as [sequence x resdiue number x one-hot amino acids]
        Model:
            [1] Linear encoder converts one-hot amino acids to dense representation
            [2] Multiple hidden layers with ReLU activation functions
            [3] Linear regression layer predicts binding value for each sample
    """

    def __init__(self, sequences, data, amino_acids='ADEFGHKLNPQRSVWYX', amino_embedder_nodes=10,
                 batch_size=100, chemical_embedder=False, evaluate_model=False, fit_sample=False, hidden_layers=2,
                 hidden_nodes=100, layer_freeze=0, learn_rate=0.001, log_shift=100, random_sequence_shifts=True,
                 saturation_threshold=0.99, save_predictions=False, save_weights=False, sequence_embedder_nodes=False,
                 train_fraction=0.9, train_steps=50000, transfer_learning=False, weight_folder='fits'):
        """Parameter and file structure initialization

        Arguments:
            sequences {str} -- path to sequences with optional train and test split in second column
            data {str} -- path to binding data values
        
        Keyword Arguments:
            amino_acids {str} -- amino acid letter codes (default: {'ADEFGHKLNPQRSVWY'})
            amino_embedder_nodes {int} -- number of features to describe amino acids (default: {10})
            batch_size {int} -- batch size for training (default: {100})
            chemical_embedder {str} -- path to amino acid chemical embeddings (default: {False})
            evaluate_model {str} -- path to 'Model.pth' to evaluate model (default: {False})
            fit_sample {int} -- sample number to fit from data (default: {False})
            hidden_layers {int} -- number of hidden layers in neural network (default: {2})
            hidden_nodes {int} -- number of nodes per hidden layer of neural network (default: {100})
            layer_freeze {str} -- number of layers to freeze for transfer learning (default: {0})
            learn_rate {float} -- magnitude of gradient descent step (default: {0.001})
            log_shift {int} -- value to shift data before applying logarithm (default: {100})
            random_sequence_shifts {bool} -- randomly shift input sequence positions (default: {True})
            saturation_threshold {float} -- saturation level to exclude from training (default: {0.99})
            save_predictions {bool} -- save predictions to file (default: {False})
            save_weights {bool} -- save weights to file (default: {False})
            sequence_embedder_nodes {int} -- sequence embedding size (default: {False})
            train_fraction {float} -- fraction of non-saturated data for training (default: {0.9})
            train_steps {int} -- number of training steps (default: {50000})
            transfer_learning {str} -- path to 'Model.pth' for transfer learning (default: {False})
            weight_folder {str} -- directory name to save weights and biases (default: {'fits'})
        """
        # Initialize input paths
        self.sequences = sequences
        self.data = data

        # Initialize parameters
        self.amino_acids = amino_acids
        self.amino_embedder_nodes = amino_embedder_nodes
        self.batch_size = batch_size
        self.chemical_embedder = chemical_embedder
        self.evaluate_model = evaluate_model
        self.fit_sample = fit_sample
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.layer_freeze = layer_freeze
        self.learn_rate = learn_rate
        self.log_shift = log_shift
        self.random_sequence_shifts = random_sequence_shifts
        self.saturation_threshold = saturation_threshold
        self.save_predictions = save_predictions
        self.save_weights = save_weights
        self.sequence_embedder_nodes = sequence_embedder_nodes
        self.train_fraction = train_fraction
        self.train_steps = train_steps
        self.transfer_learning = transfer_learning
        self.weight_folder = weight_folder
        
        # Store parameter settings
        self.settings = {key: value for key, value in locals().items() if key != 'self'}

        # Read chemical embeddings
        self.chemical_embedder = self.chemical_embedder if self.chemical_embedder else []
        if len(self.chemical_embedder):
            amino_chemistry = pd.read_csv(self.chemical_embedder, delimiter='\t', header=0, index_col=0, skiprows=1)
            amino_chemistry = torch.from_numpy(amino_chemistry.loc[list(self.amino_acids)].values).float()
            self.chemical_embedder = amino_chemistry

            # Assert match between amino embedder nodes and chemical features
            chemical_features = amino_chemistry.shape[1]
            assert self.amino_embedder_nodes == chemical_features, f'Set amino_embedder_nodes to {chemical_features}!'            

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
            filename = os.path.basename(self.sequences).split('.')[0]
            self.run_folder = date_folder + '/Run' + str(old_runs + 1) + '-' + filename
            os.makedirs(self.run_folder)
    
    def convert_sequences_to_tokens(self, sequences, max_length, padding='right'):
        """Convert sequence strings to token representation

        Args:
            sequences (series): Amino acid peptide sequences
            max_length (int): Longest peptide sequence length
            padding (str, optional): Side to pad sequences to max length. Defaults to 'right'.

        Returns:
            tensor: Token representation of sequences
        """

        # Pad sequences with X to max length
        sequences_padded = sequences.str.pad(width=max_length, side=padding, fillchar='X')

        # Tokenize sequences
        amino_acid_indices = {aa: i for i, aa in enumerate(self.amino_acids)}
        sequences_tokenized = sequences_padded.apply(lambda x: [amino_acid_indices[aa] for aa in x]).to_list()

        return torch.tensor(sequences_tokenized)
        
    def convert_tokens_to_one_hot(self, tokens):
        """Convert sequence tokens to one-hot representation

        Args:
            tokens (tensor): Token representation of sequences

        Returns:
            tensor: One-hot representation of sequences
        """
        # Randomly shift token residue positions
        if self.random_sequence_shifts:
            shifts = (((tokens == len(self.amino_acids) - 1).sum(dim=1) + 1) * torch.rand(len(tokens))).int()
            rows, columns = tokens.shape
            new_indices = ((torch.arange(columns).view((columns, 1)).repeat((1, rows)) - shifts) % columns).T
            tokens = torch.gather(tokens, 1, new_indices)
            
        return functional.one_hot(tokens, num_classes=len(self.amino_acids)).float()

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

        # Import sequences and data
        sequences = pd.read_csv(self.sequences, header=None)
        data = np.log10(pd.read_csv(self.data, header=None).values + self.log_shift)

        # Extract single sample from data
        data = data[:, sample-1:sample] if self.fit_sample else data

        # Extract train test split assignments from sequences
        train_test_split = sequences.iloc[:, 1].tolist() if len(sequences.columns) > 1 else []
        sequences = sequences.iloc[:, 0]

        # Clean sequences
        sequences = sequences.str.strip()

        # Remove trailing GSG
        if sum(sequences.str[-3:] == 'GSG') / len(sequences) > 0.99:
            sequences = sequences.str[:-3]
        
        # Tokenize sequences
        max_length = int(sequences.str.len().max())
        sequence_tokens = self.convert_sequences_to_tokens(sequences, max_length)

        # Randomly generate split between train and test sets if not specified
        if len(train_test_split) == 0:

            # Exclude saturated binding values from train set
            saturation = self.saturation_threshold * np.ptp(data) + np.min(data)

            # Assign train and test set indices
            nonsaturated = np.where(data.max(axis=1) <= saturation)[0]
            train_size = int(self.train_fraction * len(nonsaturated))
            train_split = np.random.choice(nonsaturated, train_size, replace=False)
            train_test_split = np.ones(len(sequences), dtype=int)
            train_test_split[train_split] = 0

        # Split into train and test sets
        train_sequences = sequence_tokens[[x == 0 for x in train_test_split]]
        train_data = np.copy(data[[x == 0 for x in train_test_split]])
        test_sequences = sequence_tokens[[x == 1 for x in train_test_split]]
        test_data = np.copy(data[[x == 1 for x in train_test_split]])

        # Find bin indices for uniformly distributed batch gradient descent
        train_sequences = train_sequences[train_data.max(axis=1).argsort()]
        train_data = train_data[train_data.max(axis=1).argsort()]
        bin_data = np.linspace(train_data.max(axis=1).min(), train_data.max(axis=1).max(), self.batch_size)
        bin_ind = [np.argmin(np.abs(x - train_data.max(axis=1))) for x in bin_data]
        bin_ind = np.append(bin_ind, len(train_data))

        # Convert to PyTorch variable tensors
        train_data = torch.from_numpy(train_data).float()
        test_data = torch.from_numpy(test_data).float()


        # Neural network architecture
        class Architecture(nn.Module):

            def __init__(self, amino_embeddings, chemical_embeddings,
                         inputs, layers, nodes, outputs, sequence_embeddings):
                super().__init__()

                # Network parameters
                self.amino_embeddings = amino_embeddings
                self.chemical_embeddings = chemical_embeddings
                self.inputs = inputs
                self.nodes = nodes
                self.outputs = outputs
                self.sequence_embeddings = sequence_embeddings if sequence_embeddings else nodes

                # Amino acid embedder layer
                amino_input = amino_embeddings if len(self.chemical_embeddings) else inputs
                amino_bias = bool(len(self.chemical_embeddings))
                self.amino_embedder_layer = nn.Linear(amino_input, amino_embeddings, bias=amino_bias)
                
                # Hidden layers
                hidden_input = max_length * amino_embeddings if amino_embeddings else max_length * inputs
                self.hidden_layers = nn.ModuleList([nn.Linear(hidden_input, nodes, bias=True)])
                self.hidden_layers.extend([nn.Linear(nodes, nodes, bias=True) for _ in range(layers - 2)])

                # Sequence embedder layer
                self.hidden_layers.append(nn.Linear(nodes, self.sequence_embeddings, bias=True))

                # Output layer
                self.output_layer = nn.Linear(self.sequence_embeddings, outputs, bias=True)

            def forward(self, seqs):
                if self.amino_embeddings:
                    seqs = torch.mm(seqs, self.chemical_embeddings) if len(self.chemical_embeddings) else seqs
                    seqs = self.amino_embedder_layer(seqs)
                    seqs = seqs.view(-1, max_length * self.amino_embeddings)
                for x in self.hidden_layers:
                    seqs = functional.relu(x(seqs))
                return self.output_layer(seqs)

        net = Architecture(amino_embeddings=self.amino_embedder_nodes, chemical_embeddings=self.chemical_embedder,
                           inputs=len(self.amino_acids),
                           layers=self.hidden_layers, nodes=self.hidden_nodes, outputs=train_data.shape[1],
                           sequence_embeddings=self.sequence_embedder_nodes)
        print('\nARCHITECTURE:')
        print(net)

        # Loss function and optimizer
        losses = []
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.learn_rate)

        # Transfer learn from trained model
        if self.transfer_learning:

            # Load pretrained weights
            net.load_state_dict(torch.load(self.transfer_learning))

            # Freeze initial layer weights and biases
            weight_bias_freeze = 2 * self.layer_freeze
            weight_bias_freeze -= 1 if self.amino_embedder_nodes else 0
            for parameter in net.parameters():
                if weight_bias_freeze <= 0:
                    break
                else:
                    parameter.requires_grad = False
                    weight_bias_freeze -= 1

        # Evaluate trained model
        if self.evaluate_model:
            net.load_state_dict(torch.load(self.evaluate_model))

        # Train model
        else:

            # Record best network weights and loss
            print('\nTRAINING:')
            for i in range(self.train_steps + 1):

                # Select indices for training
                train_ind = [np.random.randint(bin_ind[i], bin_ind[i + 1] + 1) for i in range(len(bin_ind) - 1)]
                train_ind[-1] = train_ind[-1] - 1

                # Calculate loss
                train_out = net(self.convert_tokens_to_one_hot(train_sequences[train_ind]))
                loss = loss_function(train_out, train_data[train_ind])

                # Weight optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Report training progress every 100 steps
                if i % 100 == 0:

                    # Select 1000 training and test indices for progress report
                    train_batch = random.sample(range(train_sequences.shape[0]), 1000)
                    test_batch = random.sample(range(test_sequences.shape[0]), 1000)

                    # Train and test binding predictions
                    train_prediction = net(self.convert_tokens_to_one_hot(train_sequences[train_batch]))
                    test_prediction = net(self.convert_tokens_to_one_hot(test_sequences[test_batch]))

                    # Record train and test losses
                    train_loss = loss_function(train_prediction, train_data[train_batch])
                    test_loss = loss_function(test_prediction, test_data[test_batch])
                    losses.append((i, train_loss, test_loss))

                    train_accuracy = torch.abs(train_prediction.flatten() - train_data[train_batch].flatten())
                    train_accuracy = len(torch.nonzero(train_accuracy < 0.2)) / len(train_accuracy)
                    
                    test_accuracy = torch.abs(test_prediction.flatten() - test_data[test_batch].flatten())
                    test_accuracy = len(torch.nonzero(test_accuracy < 0.2)) / len(test_accuracy)

                    # Report train and test accuracies
                    print(f'Step {i:5d}: train|test accuracy - {train_accuracy:.2f}|{test_accuracy:.2f}')

        # Determine correlation coefficients for optimized neural network
        train_batch = len(train_sequences) if len(train_sequences) < 10000 else 10000
        train_batch = random.sample(range(train_sequences.shape[0]), train_batch)
        train_prediction = net(self.convert_tokens_to_one_hot(train_sequences[train_batch])).data.numpy()
        train_real = train_data[train_batch].data.numpy()
        train_correlation = np.corrcoef(train_real.flatten(), train_prediction.flatten())[0, 1]

        test_batch = len(test_sequences) if len(test_sequences) < 10000 else 10000
        test_batch = random.sample(range(test_sequences.shape[0]), test_batch)
        test_prediction = net(self.convert_tokens_to_one_hot(test_sequences[test_batch])).data.numpy()
        test_real = test_data[test_batch].data.numpy()
        test_correlation = np.corrcoef(test_real.flatten(), test_prediction.flatten())[0, 1]
        print(f'Correlation Coefficient: train|test - {train_correlation:.3f}|{test_correlation:.3f}')

        # Extract weights from model
        if self.amino_embedder_nodes:
            amino_layer = net.amino_embedder_layer.weight.data.transpose(0, 1).numpy()
        hidden_layer = [[x.weight.data.transpose(0, 1).numpy(),
                        x.bias.data.numpy()]for x in net.hidden_layers]
        output_layer = [net.output_layer.weight.data.transpose(0, 1).numpy(),
                        net.output_layer.bias.data.numpy()]

        # Turn off interactive mode
        plt.ioff()

        # Scatter plot of predicted vs real
        fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
        ax1[0].scatter(train_real, train_prediction, s=1, color='b')
        ax1[0].plot([train_real.min(), train_real.max()],
                    [train_real.min(), train_real.max()], color='k')
        ax1[0].set_xlabel('Real', fontsize=15)
        ax1[0].set_ylabel('Prediction', fontsize=15)
        ax1[0].set_title(f'Train Correlation: {train_correlation:.3f}', fontsize=15)

        ax1[1].scatter(test_real, test_prediction, s=1, color='b')
        ax1[1].plot([test_real.min(), test_real.max()],
                    [test_real.min(), test_real.max()], color='k')
        ax1[1].set_xlabel('Real', fontsize=15)
        ax1[1].set_ylabel('Prediction', fontsize=15)
        ax1[1].set_title(f'Test Correlation: {test_correlation:.3f}', fontsize=15)

        # Amino acid similarity matrix
        if self.amino_embedder_nodes and not len(self.chemical_embedder):
            amino_similar = np.linalg.norm(amino_layer, axis=1)
            amino_similar = np.array([self.amino_embedder_nodes * [magnitude] for magnitude in amino_similar])
            amino_similar = np.dot((amino_layer / amino_similar),
                                   np.transpose(amino_layer / amino_similar))
            fig2 = plt.matshow(amino_similar, cmap='coolwarm')
            plt.xticks(range(len(self.amino_acids)), self.amino_acids)
            plt.yticks(range(len(self.amino_acids)), self.amino_acids)
            plt.colorbar()
            plt.clim(-1, 1)

        # Save run to file
        if self.save_weights:

            # Create path to new sample folder
            directory = f'{self.run_folder}/Sample{str(abs(sample))}'
            os.makedirs(directory)

            # Save train test split
            with open(f'{directory}/TrainTestSplit.txt', 'w') as f:
                f.writelines(f'{x},{y}\n' for x, y in zip(sequences, train_test_split))

            # Save weights and biases to csv files
            if self.amino_embedder_nodes:
                np.savetxt(f'{directory}/W1.txt', amino_layer, delimiter=',')
            if len(self.chemical_embedder):
                np.savetxt(f'{directory}/B1.txt', net.amino_embedder_layer.bias.data.numpy(), delimiter=',')
            for (m, n) in enumerate(hidden_layer):
                np.savetxt(f'{directory}/W{str(m + 2)}.txt', n[0], delimiter=',')
                np.savetxt(f'{directory}/B{str(m + 2)}.txt', n[1], delimiter=',')
            np.savetxt(f'{directory}/WF.txt', output_layer[0], delimiter=',')
            np.savetxt(f'{directory}/BF.txt', output_layer[1], delimiter=',')

            # Save correlation coefficient to file
            with open(f'{directory}/Correlation.txt', 'w') as f:
                f.write(f'Train: {train_correlation}\n')
                f.write(f'Test: {test_correlation}')
            
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
            if self.amino_embedder_nodes and not len(self.chemical_embedder):
                fig2.figure.savefig(f'{directory}/Similarity.png', bbox_inches='tight')
                plt.close()

            # Save model
            torch.save(net.state_dict(), f'{directory}/Model.pth')

            # Save log file of most recent fit
            with open(os.path.join(self.weight_folder, f'{self.weight_folder}.log'), 'w') as f:
                f.write(f'{datetime.datetime.now()},{directory}\n')
        
        # Save all predictions
        if self.save_predictions:
            
            # Save all predictions
            with open(f'{directory}/Predictions.txt', 'w') as f:
                for i in range(0, len(sequence_tokens), 1000):
                    predictions = net(self.convert_tokens_to_one_hot(sequence_tokens[i:i+1000])).data.numpy()
                    np.savetxt(f, 10**(predictions) - self.log_shift, fmt='%.5f', delimiter=',')

        # Show figures
        else:
            plt.show()


#=== RUN PEPTIDE-ARRAY-ML ===#
if __name__ == '__main__':
    samples = pd.read_csv(parameter_settings['data'], header=None).values.shape[1]
    neural_network = NeuralNetwork(**parameter_settings)
    pool = Pool()
    pool.map(neural_network.fit, range(1, samples + 1 if parameter_settings['fit_sample'] else 2))
