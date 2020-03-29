# Plot correlation between replicates for the same target

# Import modules
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Preallocate target data
data = {}

# Read HT-V13 peptide array data
peptide_array = pd.read_excel('data/HT-V13.xlsx', index_col=0, skiprows=1,
                              usecols=[0, 2, 3, 6, 7, 10, 11])
for i, target in enumerate(['Diaphorase', 'Ferredoxin', 'FNR']):
    data[target] = np.log10(peptide_array.iloc[:, 2*i:2*i+2].values + 100)

# Read HT-V13 peptide array data
peptide_array = pd.read_excel('data/CIMw189-s9.xlsx', index_col=0, skiprows=6,
                              usecols=[2, 4, 5, 9, 10, 13, 14, 16, 17, 25, 26])
for i, target in enumerate(['PDL1', 'PD1', 'TNFα', 'TNFR', 'Fc']):
    data[target] = np.log10(peptide_array.iloc[:, 2*i:2*i+2].values + 100)

# Plot self correlations
fig, ax = plt.subplots(3, 3)
# targets = ['Diaphorase', 'Ferredoxin', 'FNR', 'PD1', 'PDL1', 'TNFα', 'TNFR', 'Transferrin', 'Fc']
targets = ['Diaphorase', 'Ferredoxin', 'FNR', 'PD1', 'PDL1', 'TNFα', 'TNFR', 'Fc']
for i, target in enumerate(targets):

    # Calculate point density
    x, y = data[target][:, 0], data[target][:, 1]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    index = z.argsort()
    x, y, z = x[index], y[index], z[index]
    padding = 0.05 * (max(x) - min(x))
    limits = [min(x) - padding, max(x) + padding]

    # Plot scatter plot colored by density
    ax[i//3, i%3].scatter(x, y, c=z, s=2, edgecolor='')
    ax[i//3, i%3].plot(limits, limits, 'k')
    ax[i//3, i%3].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[i//3, i%3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[i//3, i%3].set_xlim(limits)
    x_min, x_max = ax[i//3, i%3].get_xlim()
    y_min, y_max = ax[i//3, i%3].get_ylim()
    ax[i//3, i%3].text(x_min + 0.1 * (x_max - x_min), y_max - 0.15 * (y_max - y_min),
                       target.split('.')[0], fontname='Arial', fontsize=15)

# Save figure
ax[2, 1].set_xlabel(r'Log$_{10}$(Measured Fluorescence Counts)', fontname='Arial', fontsize=20)
ax[1, 0].set_ylabel(r'Log$_{10}$(Predicted Fluorescence Counts)', fontname='Arial', fontsize=20)
fig.set_size_inches((10, 10), forward=False)
plt.savefig('figures/self_correlations.jpg', dpi=300)