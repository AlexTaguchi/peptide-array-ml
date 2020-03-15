# peptide-array-ml
Neural network modeling of peptide sequence binding data

### Installation
1) Download and install the Python 3 version of Anaconda (https://www.anaconda.com/download/)
2) Install PyTorch with one of the following commands in a terminal/command prompt (https://pytorch.org):
   - Mac: `conda install pytorch torchvision -c pytorch`
   - Linux/Windows: `conda install pytorch torchvision cpuonly -c pytorch`
3) Download this GitHub repository

### Scripts
- **fit_targets.py**: Model all nine protein targets with a neural network. The parameter settings (`Parameters.txt`), trained models (`Model.pth`), predicted vs real correlation plots (`Correlation.png`), and amino acid similarity matrix plots (`Similarity.png`) are all saved along with other metadata in `fits/`.
- **multifit_target.py**: Fit a neural network multiple times to the same target. This is useful for estimating the variability in model performance and predictions.
- **extrapolate_binding.py**: Train a neural network on weaker binding values, and then use it to predict stronger binding values not previously seen by the model.
- **plot_specificity.py**: Plot measured and predicted binding specificity between two targets.
- **plot_loss.py**: Plot loss during model training.
- **plot_distributions.py**: Plot length and amino acid distributions of the peptide arrays.
