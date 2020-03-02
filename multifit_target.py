# Model single target multiple times in parallel with neural network

# Import modules
from multiprocessing import Pool
from peptide_array_ml import NeuralNetwork

# Fit target data
nn = NeuralNetwork(filename=f'data/FNR.csv', weight_save=True)
pool = Pool()
pool.map(nn.fit, range(1, 4))