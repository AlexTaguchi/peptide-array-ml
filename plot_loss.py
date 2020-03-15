# Plot train and test loss

# Import modules
import datetime
import matplotlib.pyplot as plt
import os
from peptide_array_ml import NeuralNetwork
import re

# Set up figure for plotting
fig, ax = plt.subplots(3, 3)

# Fit target data
targets = ['Diaphorase', 'Ferredoxin', 'FNR', 'PD1', 'PDL1', 'TNFa', 'TNFR', 'Transferrin', 'Fc']
for i, target in enumerate(targets):

    # Train model
    nn = NeuralNetwork(filename=f'data/{target}.csv', weight_save=True, train_steps=50000)
    nn.fit()

    # Read in train and test loss
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    runs = [x for x in os.listdir(f'fits/{date}') if 'Run' in x]
    run = sorted(runs, key=lambda x: int(re.findall(r'\d+', x)[0]))[-1]
    loss = [[], []]
    with open(os.path.join('fits', date, run, 'Sample1/Loss.txt'), 'r') as lines:
        for line in lines:
            line = line.split()[-1].split('|')
            loss[0].append(float(line[0]))
            loss[1].append(float(line[1]))
    
    # Plot train and test loss
    ax[i//3, i%3].plot(list(range(0, 50000, 100)), loss[0][1:])
    ax[i//3, i%3].plot(list(range(0, 50000, 100)), loss[1][1:])
    ax[i//3, i%3].set_title(target.split('.')[0], fontsize=20)

# Show plots
ax[1, 0].set_ylabel('Mean Squared Error Loss', fontsize=18)
ax[2, 1].set_xlabel('Training Step', fontsize=18)
fig.subplots_adjust(left=0.1, right=0.95, hspace=0.3)
plt.show()
