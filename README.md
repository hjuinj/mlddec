# mlddec
Random forest models for predicting atomic partial charges, fitted to DFT + DDEC6 calculations.

Modified code based on [our publication](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00663).

## Installation
This package only supports Python 3.5 or above.

The mlddec packages is built on various packages that is conda-installable. Therefore, either anaconda or miniconda is needed as prerequisites.

Once conda is available, execute the following commands to install all essential dependencies:
```bash
conda install -c conda-forge rdkit
conda install -c conda-forge tqdm
conda install scikit-learn, numpy
conda install -c conda-forge matplotlib  #only if you want to visualise the molecule with the charges
```

After all dependencies are installed, navigate to a directory where you wish to clone this repository and execute:
```
git clone https://github.com/rinikerlab/mlddec.git
cd mlddec
python setup.py install
```


## Usage
```python
import mlddec


#Load all the machine-learned models for a given epsilon, currently only epsilon of 4 or 78 is available. The latter gives more polar charges
epsilon = 4
models  = mlddec.load_models(epsilon)

#If you want to validate the installation is correct, you can run the following:
mlddec.validate_models(models, epsilon)

#To charge a molecule, run the following:
from rdkit import Chem
mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))

charges = mlddec.get_charges(mol, models)
mlddec.add_charges_to_mol(mol, charges=charges)

#You can look at the molecule in 2D with the assigned charges with:
mlddec.visualise_charges(mol)

#Once you charged all the molecules, you should unload the models as they consume quite some memory
del models
```


## Maintainer
Shuzhe Wang
