# Plot comparison of measured and predicted binding specificity between two targets

# Import modules
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from peptide_array_ml import NeuralNetwork
import re

# Set targets for specificity comparison
targets = ['Diaphorase.csv', 'FNR.csv']

# Import binding datasets
data = [pd.read_csv(f'data/{target}', header=None)[1].values for target in targets]
data = np.log10(np.vstack(data) + 100)

# Randomly split into train and test sets
train_test_split = np.random.choice(len(data[0]), 90000, replace=False)
train_test_split = [0 if i in train_test_split else 1 for i in range(len(data[0]))]

# Fit model to targets
specificity = []
for target in targets:
    nn = NeuralNetwork(filename=f'data/{target}', train_test_split=train_test_split, weight_save=True)
    nn.fit()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    runs = [x for x in os.listdir(f'fits/{date}') if 'Run' in x]
    run = sorted(runs, key=lambda x: int(re.findall(r'\d+', x)[0]))[-1]
    nn.evaluation_mode = os.path.join('fits', date, run, 'Sample1/Model.pth')
    specificity.append(nn.fit()[2:])

# Plot measured and predicted target specificity
fig, ax = plt.subplots(1, 2)
colors = specificity[0][0] - specificity[1][0]
ax[0].scatter(specificity[0][0], specificity[1][0], 3, c=colors)
ax[1].scatter(specificity[0][1], specificity[1][1], 3, c=colors)
ax[0].set_xlabel(targets[0].split('.')[0])
ax[1].set_xlabel(targets[0].split('.')[0])
ax[0].set_ylabel(targets[1].split('.')[0])
ax[1].set_ylabel(targets[1].split('.')[0])
ax[0].set_title('Measured')
ax[1].set_title('Predicted')
plt.show()
