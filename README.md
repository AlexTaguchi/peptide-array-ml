# peptide-array-ml
Neural network modeling of peptide sequences and binding data

### Installation
1) Download and install the Anaconda distribution of Python 3 (https://www.anaconda.com/download/)
2) Install PyTorch with one of the following commands in a terminal/command prompt (https://pytorch.org):
   - Mac: `conda install pytorch torchvision -c pytorch`
   - Linux/Windows: `conda install pytorch torchvision cpuonly -c pytorch`
3) Download this GitHub repository

### Running on Command Line
#### Call `__init__.py` with Python and provide custom arguments file:
`python peptide_array_ml/__init__.py peptide_array_ml/arguments.txt`

### 2020 Combinatorial Science Publication
Taguchi, A. T., Boyd, J., Diehnelt, C. W., Legutki, J. B., Zhao, Z-G., and Woodbury, N. W. (2020) *Combinatorial Science*, **22** (10), 500-508
DOI: 10.1021/acscombsci.0c00003
#### Scripts to reproduce figures in `publications/2020_Combinatorial_Science`:
- **fit_targets.py**: Model all nine protein targets with a neural network. The parameter settings (`Parameters.txt`), trained models (`Model.pth`), predicted vs real correlation plots (`Correlation.png`), and amino acid similarity matrix plots (`Similarity.png`) are all saved along with other metadata in `fits/`.
- **multifit_target.py**: Fit a neural network multiple times to the same target. This is useful for estimating the variability in model performance and predictions.
- **extrapolate_binding.py**: Train a neural network on weaker binding values, and then use it to predict stronger binding values not previously seen by the model.
- **plot_specificity.py**: Plot measured vs predicted binding specificity between targets.
- **plot_specificity_control.py**: Plot measured vs predicted binding specificity for replicates of the same target.
- **plot_self_correlation.py**: Plot correlation between replicates of the same target.
- **plot_loss.py**: Plot loss during model training.
- **plot_distributions.py**: Plot length and amino acid distributions of the peptide arrays.

### License
The use of these algorithms is protected under intellectual property filed by Arizona State University
