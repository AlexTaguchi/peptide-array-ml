# Plot length and amino acid distributions of peptide arrays

# Import modules
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

# Set up figures
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots(1, 3)

# Set representative peptide arrays:
# 1. Transferrin - Transferrin
# 2. Diaphorase - Diaphorase, Ferredoxin, FNR
# 3. Fc - Fc, PD1, PDL1, TNFa, TNFR
targets = {1: 'data/Transferrin.csv',
           2: 'data/Diaphorase.csv',
           3: 'data/Fc.csv'}

# Plot length and amino acid distributions of targets
for i, target in targets.items():

    # Import and clean sequence data
    data = pd.read_csv(target, header=None)
    data[0].replace(re.compile(f'[^ADEFGHKLNPQRSVWY]'), '', inplace=True)

    # Remove trailing GSG from sequences
    if sum(data[0].str[-3:] == 'GSG') / len(data) > 0.9:
        data[0] = data[0].str[:-3]

    # Plot length distribution
    lengths = pd.Series({position: 0 for position in range(3, 14)})
    for index, value in data[0].str.len().value_counts().sort_index().iteritems():
        lengths[index] = value
    lengths /= sum(lengths)
    ax1.plot(lengths.index, lengths.values)
    ax1.scatter(lengths.index, lengths.values)

    # Count amino acid types at each position
    amino_dist = data[0].str.split('', expand=True)
    amino_dist = [amino_dist[col].value_counts() for col in amino_dist]
    amino_dist = pd.concat(amino_dist, axis=1, sort=True).iloc[1:, 1:-1].fillna(0)

    # Divide counts by total number of sequences
    amino_dist /= len(data)

    # Plot amino acid distribution
    i -= 1
    amino_dist = amino_dist.transpose()
    amino_dist.loc[:, :'P'].plot(ax=ax2[i])
    amino_dist.loc[:, 'Q':].plot(ax=ax2[i], linestyle='dashed')
    ax2[i].set_title(f'Peptide Array {i + 1}', fontsize=17)
    ax2[i].set_ylim([-0.01, 0.5])

# Finalize length distribution plot
ax1.set_xlabel('Sequence Length', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
fig1.legend(['Peptide Array 1','Peptide Array 2','Peptide Array 3'],
            loc='upper left', bbox_to_anchor=(0.15, 0.85))

# Finalize amino acid distribution plots
ax2[0].set_ylabel('Frequency', fontsize=15)
ax2[0].set_xticks(range(1, 12, 2))
ax2[1].set_xticks(range(2, 13, 2))
ax2[2].set_xticks(range(2, 13, 2))
ax2[1].set_yticks([])
ax2[2].set_yticks([])
ax2[1].set_xlabel('Sequence Position', fontsize=15)
ax2[0].get_legend().remove()
ax2[1].get_legend().remove()
ax2[2].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
fig2.subplots_adjust(left=0.07, right=0.92, wspace=0.0)
plt.show()
