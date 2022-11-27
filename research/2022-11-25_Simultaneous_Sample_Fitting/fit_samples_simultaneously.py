# Model peptide array across multiple samples simultaneously

# Add project root to path
import sys
sys.path.append('../..')

# Import modules
from peptide_array_ml import NeuralNetwork

# Fit target data
nn = NeuralNetwork(sequences='../../data/DM1A_sequence.csv', data='../../data/DM1A_data.csv')
nn.fit()
