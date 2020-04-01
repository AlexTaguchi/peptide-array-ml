# Plot specificity distributions between targets

# Import modules
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import pandas as pd
from peptide_array_ml import NeuralNetwork
import re
import matplotlib.pyplot as plt

# Preallocate target data
data = {}

# Read HT-V13 peptide array data
peptide_array = pd.read_excel('data/HT-V13.xlsx', index_col=0, skiprows=1,
                              usecols=[0, 2, 3, 6, 7, 10, 11])
for i, target in enumerate(['Diaphorase', 'Ferredoxin', 'FNR']):
    data[target] = np.log10(peptide_array.iloc[:, 2*i:2*i+2].values + 100)

# Read CIMw189-s9 peptide array data
peptide_array = pd.read_excel('data/CIMw189-s9.xlsx', index_col=0, skiprows=6,
                              usecols=[2, 4, 5, 9, 10, 13, 14, 16, 17, 25, 26])
for i, target in enumerate(['PDL1', 'PD1', 'TNFα', 'TNFR', 'Fc']):
    data[target] = np.log10(peptide_array.iloc[:, 2*i:2*i+2].values + 100)

# Plot binding specificities for Diaphorase, FNR, and Ferredoxin
fig, ax = plt.subplots(3, 1)
target_group = ['Diaphorase', 'FNR', 'Ferredoxin']
for i, target_pair in enumerate(combinations(target_group, 2)):

    # Calculate self differences
    target_1 = data[target_pair[0]][:, 0] - data[target_pair[0]][:, 1]
    target_2 = data[target_pair[1]][:, 0] - data[target_pair[1]][:, 1]

    # Calculate specificity differences
    specificity = [pd.read_csv(f'data/{target}.csv', header=None)[1].values for target in target_pair]
    specificity = np.log10(np.vstack(specificity) + 100)
    specificity = specificity[0, :] - specificity[1, :]

    # Center distributions on the median
    target_1 -= np.median(target_1)
    target_2 -= np.median(target_2)
    specificity -= np.median(specificity)

    # Take the absolute value
    target_1 = np.abs(target_1)
    target_2 = np.abs(target_2)
    specificity = np.abs(specificity)

    # Compute histograms
    x_axis = np.arange(0, 1, 0.01)
    target_1, _ = np.histogram(target_1, x_axis, density=True)
    target_2, _ = np.histogram(target_2, x_axis, density=True)
    specificity, _ = np.histogram(specificity, x_axis, density=True)

    # Record binding specificity at half height
    print(f'{target_pair[0].upper()}-{target_pair[1].upper()}')
    print(f'- {target_pair[0]}: {x_axis[np.argmin(np.abs(target_1 - 0.5 * max(target_1)))]}')
    print(f'- {target_pair[1]}: {x_axis[np.argmin(np.abs(target_2 - 0.5 * max(target_2)))]}')
    print(f'- Specificity: {x_axis[np.argmin(np.abs(specificity - 0.5 * max(specificity)))]}')

    # Plot line histograms
    ax[i].plot(x_axis[:-1], target_1)
    ax[i].plot(x_axis[:-1], target_2)
    ax[i].plot(x_axis[:-1], specificity)
    ax[i].set_yticks([])
    if i != 2:
        ax[i].set_xticks([])
    ax[i].legend([f'{target_pair[0]}', f'{target_pair[1]}',
                  f'{target_pair[0]}-{target_pair[1]}'], loc='center right')
    ax[i].set_title(f'{target_pair[0]}-{target_pair[1]}', x=0.5, y=0.8, fontsize=18)

# Save plots
plt.subplots_adjust(hspace=0)
ax[2].set_xlabel('Binding Specificity', fontname='Arial', fontsize=15)
ax[1].set_ylabel('Density', fontname='Arial', fontsize=15)
fig.set_size_inches((5, 10), forward=False)
plt.savefig('figures/specificity_distribution_HT-V13.jpg', dpi=300)

# Plot binding specificities for PD1, PDL1, TNFR, and TNFa
fig, ax = plt.subplots(3, 2)
target_group = ['PD1', 'PDL1', 'TNFR', 'TNFα']
for i, target_pair in enumerate(combinations(target_group, 2)):

    # Calculate self differences
    target_1 = data[target_pair[0]][:, 0] - data[target_pair[0]][:, 1]
    target_2 = data[target_pair[1]][:, 0] - data[target_pair[1]][:, 1]

    # Calculate specificity differences
    specificity = [pd.read_csv(f'data/{target if target != "TNFα" else "TNFa"}.csv',
                               header=None)[1].values for target in target_pair]
    specificity = np.log10(np.vstack(specificity) + 100)
    specificity = specificity[0, :] - specificity[1, :]

    # Center distributions on the median
    target_1 -= np.median(target_1)
    target_2 -= np.median(target_2)
    specificity -= np.median(specificity)

    # Take the absolute value
    target_1 = np.abs(target_1)
    target_2 = np.abs(target_2)
    specificity = np.abs(specificity)

    # Compute histograms
    x_axis = np.arange(0, 1, 0.01)
    target_1, _ = np.histogram(target_1, x_axis, density=True)
    target_2, _ = np.histogram(target_2, x_axis, density=True)
    specificity, _ = np.histogram(specificity, x_axis, density=True)

    # Record binding specificity at half height
    print(f'{target_pair[0].upper()}-{target_pair[1].upper()}')
    print(f'- {target_pair[0]}: {x_axis[np.argmin(np.abs(target_1 - 0.5 * max(target_1)))]}')
    print(f'- {target_pair[1]}: {x_axis[np.argmin(np.abs(target_2 - 0.5 * max(target_2)))]}')
    print(f'- Specificity: {x_axis[np.argmin(np.abs(specificity - 0.5 * max(specificity)))]}')

    # Plot line histograms
    ax[i%3, i//3].plot(x_axis[:-1], target_1)
    ax[i%3, i//3].plot(x_axis[:-1], target_2)
    ax[i%3, i//3].plot(x_axis[:-1], specificity)
    ax[i%3, i//3].set_yticks([])
    if i%3 != 2:
        ax[i%3, i//3].set_xticks([])
    ax[i%3, i//3].legend([f'{target_pair[0]}', f'{target_pair[1]}',
                          f'{target_pair[0]}-{target_pair[1]}'], loc='center right')
    ax[i%3, i//3].set_title(f'{target_pair[0]}-{target_pair[1]}', x=0.5, y=0.8, fontsize=18)

# Save plots
plt.subplots_adjust(hspace=0, wspace=0)
ax[2, 0].set_xlabel('Binding Specificity', fontname='Arial', fontsize=15)
ax[2, 1].set_xlabel('Binding Specificity', fontname='Arial', fontsize=15)
ax[1, 0].set_ylabel('Density', fontname='Arial', fontsize=15)
fig.set_size_inches((10, 10), forward=False)
plt.savefig('figures/specificity_distribution_CIMw189-s9.jpg', dpi=300)
