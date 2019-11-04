from multiprocessing import Pool
import os
from peptide_array_ml import NeuralNetwork

targets = [x for x in os.listdir('data') if x[-4:] == '.csv']

for target in targets:
    nn = NeuralNetwork(filename=f'data/{target}', train_steps=50000, weight_save=True, encoder_nodes=10)
    nn.fit()
