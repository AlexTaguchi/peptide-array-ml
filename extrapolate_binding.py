# Extrapolate model trained on weaker peptide array binders to stronger binders

# Import modules
import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
from peptide_array_ml import NeuralNetwork
import re

# Set up axes for plotting
fig, ax = plt.subplots(3, 3)

# Extrapolate target data
targets = [x for x in os.listdir('data') if x.split('.')[-1] == 'csv']
for i, target in enumerate(targets):

    # Limit training set to peptides within five-fold of the weakest binder + 100
    data = pd.read_csv(f'data/{target}', header=None)
    threshold = 5 * (data[1].min() + 100)

    # Split peptide array into training (0) and test (1) sets
    train_test_split = [0 if x < threshold else 1 for x in data[1]]

    # Fit model to weakest binders on the peptide array
    nn = NeuralNetwork(filename=f'data/{target}', train_test_split=train_test_split, weight_save=True)
    nn.fit()

    # Find path to trained model
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    runs = [x for x in os.listdir(f'fits/{date}') if 'Run' in x]
    run = sorted(runs, key=lambda x: int(re.findall(r'\d+', x)[0]))[-1]
    model = os.path.join('fits', date, run, 'Sample1/Model.pth')

    # Plot model performance on train and test sets
    nn.evaluation_mode = model
    train_real, train_pred, test_real, test_pred = nn.fit()
    ax[i//3, i%3].scatter(train_real, train_pred, 1)
    ax[i//3, i%3].scatter(test_real, test_pred, 1)
    ax[i//3, i%3].set_title(target.split('.')[0])

# Show plots
ax[2, 1].set_xlabel('Log10 Measured Binding Value')
ax[1, 0].set_ylabel('Log10 Predicted Binding Value')
plt.show()
