# Model single peptide array three times in parallel

# Import modules
from multiprocessing import Pool
from peptide_array_ml import NeuralNetwork

# Fit target data
nn = NeuralNetwork(data='data/FNR.csv', weight_save=True)
pool = Pool()
pool.map(nn.fit, range(3))
