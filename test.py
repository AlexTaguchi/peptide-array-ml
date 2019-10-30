from multiprocessing import Pool
from peptide_array_ml import NeuralNetwork

nn = NeuralNetwork(train_steps=5000, weight_save=True, encoder_nodes=0)
nn.fit()
# pool = Pool()
# pool.map(nn.fit, range(1, 3))