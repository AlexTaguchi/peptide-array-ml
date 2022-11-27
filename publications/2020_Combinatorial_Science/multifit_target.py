# Model single peptide array three times in parallel

# Add project root to path
import sys
sys.path.append('../..')

# Import modules
from multiprocessing import Pool
from peptide_array_ml.legacy import NeuralNetwork2020

# Fit target data
nn = NeuralNetwork2020(data='../../data/FNR.csv', save_weights=True)
pool = Pool()
pool.map(nn.fit, range(3))
