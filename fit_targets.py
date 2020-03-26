# Model peptide array binding values with neural network

# Import modules
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
from peptide_array_ml import NeuralNetwork
from scipy.stats import gaussian_kde

# Fit target data
fig, ax = plt.subplots(3, 3)
targets = ['Diaphorase.csv', 'Ferredoxin.csv', 'FNR.csv', 'PD1.csv',
           'PDL1.csv', 'TNFa.csv', 'TNFR.csv', 'Transferrin.csv', 'Fc.csv']
for i, target in enumerate(targets):
    nn = NeuralNetwork(filename=f'data/{target}', weight_save=True)
    nn.fit()
    nn.evaluation_mode = os.path.join(nn.run_folder, 'Sample1/Model.pth')
    x, y = nn.fit()[2:]

    # Calculate point density
    xy = np.vstack([x,y])
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
                       target.split('.')[0] if target != 'TNFa.csv' else 'TNFα',
                       fontname='Arial', fontsize=15)

# Save figure
ax[2, 1].set_xlabel(r'Log$_{10}$(Measured Fluorescence Counts)', fontname='Arial', fontsize=20)
ax[1, 0].set_ylabel(r'Log$_{10}$(Predicted Fluorescence Counts)', fontname='Arial', fontsize=20)
fig.set_size_inches((10, 10), forward=False)
plt.savefig('figures/target_fits.jpg', dpi=300)
