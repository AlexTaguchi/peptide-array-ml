# peptide-array-ml
Neural network modeling of peptide sequences and binding data

### Installation
1) Download and install the Anaconda distribution of Python 3 (https://www.anaconda.com/download/)
2) Install PyTorch with one of the following commands in terminal/command prompt (https://pytorch.org):
   - Mac: `conda install pytorch torchvision -c pytorch`
   - Linux/Windows: `conda install pytorch torchvision cpuonly -c pytorch`
3) Download this GitHub repository

### Running on Command Line
#### &emsp;Call \_\_init\_\_.py with Python and provide a csv-formatted arguments file as in the following example:
    python peptide_array_ml/__init__.py peptide_array_ml/arguments.csv

### Running with Scripts
#### &emsp;See `publications` directory for examples

### Publications
1. Chowdhury, R., Taguchi, A. T., Kelbauskas, L., Stafford, P., Diehnelt, C. W., Zhao, Z-G., C. Williamson, P. C., Green, V., Woodbury, N. W. (2022) Submitted to *PLoS Computational Biology*

   Matlab scripts (`publications/2022_PLoS_Computational_Biology`):
   - **neural_network_fit.m**: Build arguments file and run neural network fitting.
   - **multi_disease_classifier.m**: Classify disease states from neural network output.

2. Taguchi, A. T., Boyd, J., Diehnelt, C. W., Legutki, J. B., Zhao, Z-G., and Woodbury, N. W. (2020) *Combinatorial Science*, **22** (10), 500-508 (DOI: 10.1021/acscombsci.0c00003)
   
   Scripts to reproduce figures (`publications/2020_Combinatorial_Science`):
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
