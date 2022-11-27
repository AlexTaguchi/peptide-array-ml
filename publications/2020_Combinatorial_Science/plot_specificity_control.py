# Plot measured vs predicted binding specificity between the same target

# Add project root to path
import sys
sys.path.append('../..')

# Import modules
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import pandas as pd
from peptide_array_ml.legacy import NeuralNetwork2020
import re
from scipy.stats import gaussian_kde

# Preallocate correlation coefficients
correlations = []

# Preallocate target data
data = {}

# Read HT-V13 peptide array data
peptide_array = pd.read_excel('../../data/HT-V13.xlsx', header=None, index_col=0,
                              skiprows=2, usecols=[0, 2, 3, 6, 7, 10, 11])
for i, target in enumerate(['Diaphorase', 'Ferredoxin', 'FNR']):
    data[target] = peptide_array.iloc[:, 2*i:2*i+2]

# Read CIMw189-s9 peptide array data
peptide_array = pd.read_excel('../../data/CIMw189-s9.xlsx', header=None, index_col=0,
                              skiprows=7, usecols=[2, 4, 5, 9, 10, 13, 14, 16, 17, 25, 26])
for i, target in enumerate(['PDL1', 'PD1', 'TNFα', 'TNFR', 'Fc']):
    data[target] = peptide_array.iloc[:, 2*i:2*i+2].drop(index='empty')

# Read CIMw174-s3 peptide array data
peptide_array = pd.read_excel('../../data/CIMw174-s3.xlsx', header=None, index_col=0,
                              skiprows=4, usecols=[1, 5, 6])
for i, target in enumerate(['Transferrin']):
    data[target] = peptide_array.iloc[:, 2*i:2*i+2]

# Plot control binding specificities for all targets
fig, ax = plt.subplots(3, 3, gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
targets = ['Diaphorase', 'Ferredoxin', 'FNR', 'PD1',
           'PDL1', 'TNFα', 'TNFR', 'Transferrin', 'Fc']
for i, target in enumerate(targets):

    # Randomly split into train and test sets
    train_test_split = np.random.choice(len(data[target].index.unique()), 90000, replace=False)
    train_test_split = [0 if i in train_test_split else 1 for i in range(len(data[target].index.unique()))]

    # Predict measurement repetition 1
    data_1 = data[target].iloc[:, :1].reset_index()
    data_1.columns = [0, 1]
    data_1 = data_1.groupby(0).mean().reset_index()
    data_1.name = target
    nn = NeuralNetwork2020(data=data_1, train_test_split=train_test_split, save_weights=True)
    nn.fit()
    nn.evaluation_mode = os.path.join(nn.run_folder, 'Sample1/Model.pth')
    measured_1, predicted_1 = nn.fit()[2:]

    # Predict measurement repetition 2
    data_2 = data[target].iloc[:, 1:].reset_index()
    data_2.columns = [0, 1]
    data_2 = data_2.groupby(0).mean().reset_index()
    data_2.name = target
    nn = NeuralNetwork2020(data=data_2, train_test_split=train_test_split, save_weights=True)
    nn.fit()
    nn.evaluation_mode = os.path.join(nn.run_folder, 'Sample1/Model.pth')
    measured_2, predicted_2 = nn.fit()[2:]

    # Calculate point density
    x, y = measured_1 - measured_2, predicted_1 - predicted_2
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    index = z.argsort()
    x, y, z = x[index], y[index], z[index]
    padding = 0.05 * (max(x) - min(x))
    limits = [min(x) - padding, max(x) + padding]

    # Record Pearson correlation coefficient
    correlation = np.corrcoef(x, y)[0, 1]
    correlations.append(f'{target}: {np.corrcoef(x, y)[0, 1]:.3f}')

    # Plot measured and predicted log binding ratios
    ax[i//3, i%3].scatter(x, y, c=z, s=2, edgecolor=['none'])
    ax[i//3, i%3].plot(limits, limits, 'k')
    ax[i//3, i%3].set_xlim(limits)
    ax[i//3, i%3].set_title(f'{target}', fontname='Arial', fontsize=15)

    # Plot correlatin coefficient
    ax[i//3, i%3].text(0.06, 0.85, f'R={correlation:.3f}', style='italic',
                       fontname='Arial', fontsize=15, transform=ax[i//3, i%3].transAxes)

# Label figures
fig.text(0.5, 0.04, r'Log$_{10}$(Measured Binding Ratios)',
         ha='center', fontname='Arial', fontsize=18)
fig.text(0.04, 0.5, r'Log$_{10}$(Predicted Binding Ratios)',
         va='center', rotation='vertical', fontname='Arial', fontsize=18)

# Save plots
fig.set_size_inches((9, 9), forward=False)
plt.savefig('figures/specificity_controls.jpg', dpi=300)

# Report Pearson correlatin coefficients
print('\nSpecificity Pearson Correlation Coefficients')
for correlation in correlations:
    print(correlation)
