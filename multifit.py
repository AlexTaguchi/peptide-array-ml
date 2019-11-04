from multiprocessing import Pool
from peptide_array_ml import NeuralNetwork

nn = NeuralNetwork(filename=f'data/FNR.csv', train_steps=5000, weight_save=True, encoder_nodes=0)
pool = Pool()
pool.map(nn.fit, range(1, 3))