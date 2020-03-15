# Model peptide array binding values with neural network

# Import modules
import os
from peptide_array_ml import NeuralNetwork

# Fit target data
targets = [x for x in os.listdir('data') if x.split('.')[-1] == 'csv']
for target in targets:
    nn = NeuralNetwork(filename=f'data/{target}', weight_save=True)
    nn.fit()
