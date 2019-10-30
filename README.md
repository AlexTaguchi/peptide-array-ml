# peptide-array-ml
Neural network modeling of peptide sequence binding data

### Installation
1) Download and install the Python 3 version of Anaconda (https://www.anaconda.com/download/)
2) Install PyTorch with one of the following commands in a terminal/command prompt (https://pytorch.org):
   - Mac: `conda install pytorch torchvision -c pytorch`
   - Linux/Windows: `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`

### Usage
Download this repository and run the following command to train the neural network on a random 90% of the sequence/binding values from the default `data/FNR.csv` dataset, and evaluate its performance on the 10% hold-out test set:
```
python predict_peptide_binding.py
```
The neural network can be trained on any other dataset by specifying its path:
```
python predict_peptide_binding.py data/Diaphorase.csv
```
For more advanced features modify the `PARAMETERS` field in lines 34-44 of `predict_peptide_binding.py`. For example,
- Save the trained model and resulting figures instead of plotting them by setting `weightSave` on line 44 to `True`.
- The one-hot amino acid representations are encoded into dense representations with the `aminoEncoder` parameter. The size of the encoded amino acid feature vector can be changed by adjusting this value.
- A measured set of chemical properties<sup>1</sup> for the amino acids can be used instead of having the neural network learn them by setting `chemParams` to `True` (note that `aminoEncoder` must be set to 3 in this case for a meaningful comparison of the model performance using the measured and learned chemical properties).
- Evaulate the performance of a trained neural network by setting `testingMode` to the path to the `Model.pth` file.
- Perform multiple training runs in parallel (for statistical averaging) by setting `multipleRuns` to a value greater than 1.

<sup>1</sup>Meiler, J.; Müller, M.; Zeidler, A.; Schmäschke, F. (2001) Generation and evaluation of dimension-reduced amino acid parameter representations by artificial neural networks. *Journal of Molecular Modeling*, 7, 360-369.