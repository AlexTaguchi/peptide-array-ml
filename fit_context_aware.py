# Model peptide array with concentration context

# Import modules
from peptide_array_ml import ContextAware

# Fit target data
nn = ContextAware(peptides='data/DM1A_sequence.csv',
                  context='data/DM1A_concentration.csv',
                  binding='data/DM1A_data.csv')
nn.fit()
