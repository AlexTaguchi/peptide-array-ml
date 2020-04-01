# Plot comparison of measured and predicted binding specificity between targets

# Import modules
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from peptide_array_ml import NeuralNetwork
import re
from scipy.stats import gaussian_kde

# Create lookup for fitting runs
directory = 'fits/2020-01-01/'
fits = {x.split('-')[1]: x for x in os.listdir(directory)}

# Set up Figures A and B with column gap
fig, ax = plt.subplots(3, 4, gridspec_kw={'hspace':0.3, 'wspace':0.25, 'width_ratios': [1, 0.2, 1, 1]})
for i in range(3):
    ax[i, 1].set_visible(False)
ax[0, 0].text(-0.3, 1.1, 'A', transform=ax[0, 0].transAxes, fontname='Arial', fontsize=35)
ax[0, 2].text(-0.3, 1.1, 'B', transform=ax[0, 2].transAxes, fontname='Arial', fontsize=35)

# Preallocate correlation coefficients
correlations = []

# Plot binding specificities for Diaphorase, FNR, and Ferredoxin
for i, target_pair in enumerate(combinations(['Diaphorase', 'FNR', 'Ferredoxin'], 2)):

    # Import binding datasets
    data = [pd.read_csv(f'data/{target}.csv', header=None)[1].values for target in target_pair]

    # Randomly split into train and test sets
    train_test_split = np.random.choice(len(data[0]), 90000, replace=False)
    train_test_split = [0 if i in train_test_split else 1 for i in range(len(data[0]))]

    # Evaluate model for target pair
    specificity = []
    for target in target_pair:
        nn = NeuralNetwork(filename=f'data/{target}.csv', train_test_split=train_test_split,
                            evaluation_mode=f'{directory}{fits[target]}/Sample1/Model.pth')
        specificity.append(nn.fit()[2:])
    
    # Calculate point density
    x, y = specificity[0][0] - specificity[1][0], specificity[0][1] - specificity[1][1]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    index = z.argsort()
    x, y, z = x[index], y[index], z[index]
    padding = 0.05 * (max(x) - min(x))
    limits = [min(x) - padding, max(x) + padding]

    # Record Pearson correlation coefficient
    correlations.append(f'{target_pair[0]}-{target_pair[1]}: {np.corrcoef(x, y)[0, 1]:.4f}')

    # Plot measured and predicted log binding ratios
    ax[i, 0].scatter(x, y, c=z, s=2, edgecolor='')
    ax[i, 0].plot(limits, limits, 'k')
    ax[i, 0].set_xlim(limits)
    ax[i, 0].set_title(f'{target_pair[0]}-{target_pair[1]}', fontname='Arial', fontsize=15)

# Plot binding specificities for PD1, PDL1, TNFR, and TNFa
for i, target_pair in enumerate(combinations(['PD1', 'PDL1', 'TNFR', 'TNFa'], 2)):

    # Import binding datasets
    data = [pd.read_csv(f'data/{target}.csv', header=None)[1].values for target in target_pair]

    # Randomly split into train and test sets
    train_test_split = np.random.choice(len(data[0]), 90000, replace=False)
    train_test_split = [0 if i in train_test_split else 1 for i in range(len(data[0]))]

    # Evaluate model for target pair
    specificity = []
    for target in target_pair:
        nn = NeuralNetwork(filename=f'data/{target}.csv', train_test_split=train_test_split,
                            evaluation_mode=f'{directory}{fits[target]}/Sample1/Model.pth')
        specificity.append(nn.fit()[2:])
    
    # Calculate point density
    x, y = specificity[0][0] - specificity[1][0], specificity[0][1] - specificity[1][1]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    index = z.argsort()
    x, y, z = x[index], y[index], z[index]
    padding = 0.05 * (max(x) - min(x))
    limits = [min(x) - padding, max(x) + padding]

    # Record Pearson correlation coefficient
    correlations.append(f'{target_pair[0]}-{target_pair[1]}: {np.corrcoef(x, y)[0, 1]:.4f}')
    
    # Plot measured and predicted target specificity
    target_pair = [target if target != 'TNFa' else 'TNFα' for target in target_pair]
    ax[i%3, i//3 + 2].scatter(x, y, c=z, s=2, edgecolor='')
    ax[i%3, i//3 + 2].plot(limits, limits, 'k')
    ax[i%3, i//3 + 2].set_xlim(limits)
    ax[i%3, i//3 + 2].set_title(f'{target_pair[0]}-{target_pair[1]}', fontname='Arial', fontsize=15)

# Label figures
fig.text(0.5, 0.04, r'Log$_{10}$(Measured Binding Ratios)',
         ha='center', fontname='Arial', fontsize=18)
fig.text(0.04, 0.5, r'Log$_{10}$(Predicted Binding Ratios)',
         va='center', rotation='vertical', fontname='Arial', fontsize=18)

# Save plots
fig.set_size_inches((10, 9), forward=False)
plt.savefig('figures/specificity_fits.jpg', dpi=300)

# Report Pearson correlatin coefficients
print('\nSpecificity Pearson Correlation Coefficients')
for correlation in correlations:
    print(correlation)