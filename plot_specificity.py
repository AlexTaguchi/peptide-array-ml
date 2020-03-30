# Plot comparison of measured and predicted binding specificity between targets

# Import modules
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from peptide_array_ml import NeuralNetwork
import re

# Create lookup for fitting runs
directory = 'fits/2020-01-01/'
fits = {x.split('-')[1]: x for x in os.listdir(directory)}

# Plot binding specificities for Diaphorase, FNR, and Ferredoxin
fig, ax = plt.subplots(3, 2)
target_group = ['Diaphorase', 'FNR', 'Ferredoxin']
for i, target_pair in enumerate(combinations(target_group, 2)):

    # Import binding datasets
    data = [pd.read_csv(f'data/{target}.csv', header=None)[1].values for target in target_pair]
    data = np.log10(np.vstack(data) + 100)

    # Randomly split into train and test sets
    train_test_split = np.random.choice(len(data[0]), 90000, replace=False)
    train_test_split = [0 if i in train_test_split else 1 for i in range(len(data[0]))]

    # Fit model to targets
    specificity = []
    for target in target_pair:
        nn = NeuralNetwork(filename=f'data/{target}.csv', train_test_split=train_test_split,
                            evaluation_mode=f'{directory}{fits[target]}/Sample1/Model.pth')
        specificity.append(nn.fit()[2:])

    # Plot measured and predicted target specificity
    colors = specificity[0][0] - specificity[1][0]
    ax[i, 0].scatter(specificity[0][0], specificity[1][0], 3, c=colors)
    ax[i, 1].scatter(specificity[0][1], specificity[1][1], 3, c=colors)
    ax[i, 0].set_xlabel(target_pair[0].split('.')[0], fontname='Arial', fontsize=15)
    ax[i, 1].set_xlabel(target_pair[0].split('.')[0], fontname='Arial', fontsize=15)
    ax[i, 0].set_ylabel(target_pair[1].split('.')[0], fontname='Arial', fontsize=15)
    ax[i, 1].set_ylabel(target_pair[1].split('.')[0], fontname='Arial', fontsize=15)

# Save plots
ax[0, 0].set_title('Measured', fontname='Arial', fontsize=20)
ax[0, 1].set_title('Predicted', fontname='Arial', fontsize=20)
fig.set_size_inches((10, 15), forward=False)
plt.savefig('figures/specificity_HT-V13.jpg', dpi=300)

# Plot binding specificities for PD1, PDL1, TNFR, and TNFa
fig, ax = plt.subplots(3, 4)
target_group = ['PD1', 'PDL1', 'TNFR', 'TNFa']
for i, target_pair in enumerate(combinations(target_group, 2)):

    # Import binding datasets
    data = [pd.read_csv(f'data/{target}.csv', header=None)[1].values for target in target_pair]
    data = np.log10(np.vstack(data) + 100)

    # Randomly split into train and test sets
    train_test_split = np.random.choice(len(data[0]), 90000, replace=False)
    train_test_split = [0 if i in train_test_split else 1 for i in range(len(data[0]))]

    # Fit model to targets
    specificity = []
    for target in target_pair:
        nn = NeuralNetwork(filename=f'data/{target}.csv', train_test_split=train_test_split,
                            evaluation_mode=f'{directory}{fits[target]}/Sample1/Model.pth')
        specificity.append(nn.fit()[2:])

    # Plot measured and predicted target specificity
    colors = 3 * (specificity[0][0] - specificity[1][0])
    target_pair = [target if target != 'TNFa' else 'TNFα' for target in target_pair]
    ax[i%3, 2*(i//3) + 0].scatter(specificity[0][0], specificity[1][0], 3, c=colors)
    ax[i%3, 2*(i//3) + 1].scatter(specificity[0][1], specificity[1][1], 3, c=colors)
    ax[i%3, 2*(i//3) + 0].set_xlabel(target_pair[0].split('.')[0], fontname='Arial', fontsize=15)
    ax[i%3, 2*(i//3) + 1].set_xlabel(target_pair[0].split('.')[0], fontname='Arial', fontsize=15)
    ax[i%3, 2*(i//3) + 0].set_ylabel(target_pair[1].split('.')[0], fontname='Arial', fontsize=15)
    ax[i%3, 2*(i//3) + 1].set_ylabel(target_pair[1].split('.')[0], fontname='Arial', fontsize=15)

# Save plots
ax[0, 0].set_title('Measured', fontname='Arial', fontsize=20)
ax[0, 1].set_title('Predicted', fontname='Arial', fontsize=20)
ax[0, 2].set_title('Measured', fontname='Arial', fontsize=20)
ax[0, 3].set_title('Predicted', fontname='Arial', fontsize=20)
fig.set_size_inches((20, 15), forward=False)
plt.savefig('figures/specificity_CIMw189-s9.jpg', dpi=300)
