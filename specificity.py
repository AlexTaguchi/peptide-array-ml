import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
import numpy as np
import os
from peptide_array_ml import NeuralNetwork

date = '2019-11-03'
targets = ['Diaphorase', 'Ferredoxin', 'FNR']


fits = {}
for target in targets:
    run = [x for x in os.listdir(f'fits/{date}') if x.split('-')[-1] == target][0]
    nn = NeuralNetwork(filename=f'data/{target}.csv',
                       evaluation_mode=f'fits/{date}/{run}/Sample1/Model.pth',
                       train_test_split=f'fits/{date}/Run6-Diaphorase/Sample1/TrainTestSplit.txt',
                       weight_save=False,
                       encoder_nodes=10)
    fits[target] = nn.fit()

sample = np.random.choice(range(len(fits['Diaphorase'][0])), 10000)

test = np.vstack((fits['Diaphorase'][0], fits['Ferredoxin'][0], fits['FNR'][0])).T
test -= np.sum(test, axis=1, keepdims=True) / 3
# unit = np.array([1, 1, 1]) / np.sqrt(3)
# doty = np.expand_dims(np.dot(test - np.array([[1, 0, 0]]), unit), 1)
# final = test - (doty * np.expand_dims(unit, 0))
# e
# final -= np.min(final, axis=0, keepdims=True)
# final /= np.max(final, axis=0, keepdims=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=45, azim=45)
# ax.scatter(fits['Diaphorase'][0], fits['Ferredoxin'][0], fits['FNR'][0], marker='o')
# ax.plot3D([3, 4], [3, 4], [3, 4], 'gray')
ax.scatter(test[:, 0], test[:, 1], test[:, 2], marker='o')
plt.show()

e
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=45, azim=45)
ax.scatter(fits['Diaphorase'][1], fits['Ferredoxin'][1], fits['FNR'][1], marker='o')
plt.show()