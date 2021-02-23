"""
# removing obsolete models including git history:
https://help.github.com/en/articles/removing-files-from-a-repositorys-history

# model persistence
https://cmry.github.io/notes/serialize
https://stackabuse.com/scikit-learn-save-and-restore-models/
https://github.com/scikit-learn/scikit-learn/issues/10319
https://stackoverflow.com/questions/20156951/how-do-i-find-which-attributes-my-tree-splits-on-when-using-scikit-learn
http://thiagomarzagao.com/2015/12/08/saving-TfidfVectorizer-without-pickles/

#############################################
#Forest to trees Extraction code:
import joblib
import glob

for model in glob.glob("*.model"):
        print(model)
        forest = joblib.load(model).estimators_
        for idx, tree in enumerate(forest):
                joblib.dump(tree, "./epsilon_4/{}_{}.model".format(model.split(".")[0], idx), compress = 9)
#############################################
"""
import rdkit
from rdkit import Chem


def validate_models(model_dict, epsilon):
    import numpy as np
    import tqdm
    try:
        smiles = np.load(
            _get_data_filename("validated_results/test_smiles.npy"),
            allow_pickle=True)
        charges = np.load(_get_data_filename(
            "validated_results/test_charges_{}.npy".format(epsilon)),
                          allow_pickle=True)
    except ValueError:
        raise ValueError("No model for epsilon value of {}".format(epsilon))

    print("Checking through molecule dataset, stops when discrepencies are observed...")
    for s,c in tqdm.tqdm(list(zip(smiles, charges))):
        charge_on_fly = get_charges(Chem.AddHs(Chem.MolFromSmiles(s)), model_dict)
        discrepency = ~np.isclose(charge_on_fly , c, atol = 0.01)
        if np.any(discrepency):
            tmp = np.where(discrepency)[0]
            print("No close match for {}, validation terminated. \n Atom Indices: {} \n Calculated Charges: {} \n Reference Charges: {} \n".format(s, tmp, np.array(charge_on_fly)[tmp], np.array(c)[tmp]))
            return


def _get_data_filename(relative_path):
    """Get the full path to one of the reference files in testsystems.
    In the source distribution, these files are in ``openforcefield/data/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.
    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the repex folder).
    """
    import os
    from pkg_resources import resource_filename
    fn = resource_filename('mlddec', os.path.join('data', relative_path))
    if not os.path.exists(fn):
        raise ValueError(
            "Sorry! %s does not exist. If you just added it, you'll have to re-install"
            % fn)
    return fn


def load_models(epsilon=4):
    # from sklearn.externals import joblib
    import joblib
    # supported elements (atomic numbers)
    # H, C, N, O, F, P, S, Cl, Br, I
    element_list = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    elementdict = {8:"O", 7:"N", 6:"C", 1:"H", \
                   9:"F", 15:"P", 16:"S", 17:"Cl", \
                   35:"Br", 53:"I"}
    #directory, containing the models
    if epsilon not in [4, 78]:
        raise ValueError("cluster_method should be one of 4 or 78 but is {}".format(epsilon))


    progress_bar = True
    try:
        import tqdm
    except ImportError:
        progress_bar = False
    print("Loading models...")
    try:
        if progress_bar:
            rf = {
                element: [
                    joblib.load(
                        _get_data_filename("epsilon_{}/{}_{}.model".format(
                            epsilon, elementdict[element], i)))
                    for i in range(100)
                ]
                for element in tqdm.tqdm(element_list)
            }

        else:
            rf = {
                element: [
                    joblib.load(
                        _get_data_filename("epsilon_{}/{}_{}.model".format(
                            epsilon, elementdict[element], i)))
                    for i in range(100)
                ]
                for element in element_list
            }

    except ValueError:
        raise ValueError("No model for epsilon value of {}".format(epsilon))
    return rf


def get_charges(mol, model_dict):
    """
    Parameters
    -----------
    mol : rdkit molecule
    model_dict : dictionary of random forest models
    """
    from rdkit import DataStructs, Chem
    from rdkit.Chem import AllChem
    import numpy as np

    num_atoms = mol.GetNumAtoms()
    if num_atoms != Chem.AddHs(mol).GetNumAtoms():
        import warnings
        warnings.warn("Have you added hydrogens to the molecule?", UserWarning)

    element_list = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

    # maximum path length in atompairs-fingerprint
    APLength = 4

    # check for unknown elements
    curr_element_list = []
    for at in mol.GetAtoms():
        element = at.GetAtomicNum()
        if element not in element_list:
            raise ValueError(
                "Error: element {} has not been parameterised".format(element))
        curr_element_list.append(element)
    curr_element_list = set(curr_element_list)

    pred_q = [0] * num_atoms
    sd_rf = [0] * num_atoms
    # loop over the atoms
    for i in range(num_atoms):
        # generate atom-centered AP fingerprint
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol,
                                                           maxLength=APLength,
                                                           fromAtoms=[i])
        arr = np.zeros(1, )
        DataStructs.ConvertToNumpyArray(fp, arr)
        # get the prediction by each tree in the forest
        element = mol.GetAtomWithIdx(i).GetAtomicNum()
        per_tree_pred = [
            tree.predict(arr.reshape(1, -1)) for tree in model_dict[element]
        ]
        # then average to get final predicted charge
        pred_q[i] = np.average(per_tree_pred)
        # and get the standard deviation, which will be used for correction
        sd_rf[i] = np.std(per_tree_pred)

    #########################
    # CORRECT EXCESS CHARGE #
    #########################

    # calculate excess charge
    deltaQ = sum(pred_q) - float(AllChem.GetFormalCharge(mol))
    charge_abs = 0.0
    for i in range(num_atoms):
        charge_abs += sd_rf[i] * abs(pred_q[i])
    deltaQ /= charge_abs
    # correct the partial charges

    return [(pred_q[i] - abs(pred_q[i]) * sd_rf[i] * deltaQ)
            for i in range(num_atoms)]


def add_charges_to_mol(mol,
                       model_dict=None,
                       charges=None,
                       property_name="PartialCharge"):
    """
    if charges is None, perform fitting using `get_charges`, for this `model_dict` needs to be provided
    """
    if type(charges) is list:
        assert mol.GetNumAtoms() == len(charges)
    elif charges is None and model_dict is not None:
        charges = get_charges(mol, model_dict)
    for i, atm in enumerate(mol.GetAtoms()):
        atm.SetDoubleProp(property_name, charges[i])
    return mol


def _draw_mol_with_property(mol, property, use_similarity_map=False, **kwargs):
    """
    http://rdkit.blogspot.com/2015/02/new-drawing-code.html

    Parameters
    ---------
    property : dict
        key atom idx, val the property (need to be stringfiable)
    """
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem

    def run_from_jupyter():
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
                return False
        except ImportError:
            return False
        return True

    AllChem.Compute2DCoords(mol)
    if not use_similarity_map:
        if rdkit.__version__ >= '2020.09.1':
            pn = 'atomNote'
        else:
            pn = 'molAtomMapNumber'
        for idx in property:
            mol.GetAtomWithIdx(idx).SetProp(pn, f"({property[idx]})")
    else:
        from rdkit.Chem.Draw import SimilarityMaps
        weights = [
            float(property.get(idx, 0)) for idx in range(mol.GetNumAtoms())
        ]

    mol = Draw.PrepareMolForDrawing(mol,
                                    kekulize=False)  #enable adding stereochem

    w = kwargs.get('width', 500)
    h = kwargs.get('height', 250)
    if run_from_jupyter():
        from IPython.display import SVG, display
        drawer = Draw.MolDraw2DSVG(w, h)
        if not use_similarity_map:
            drawer.DrawMolecule(mol)
        else:
            SimilarityMaps.GetSimilarityMapFromWeights(mol,
                                                       weights,
                                                       draw2d=drawer)
        drawer.FinishDrawing()
        display(SVG(drawer.GetDrawingText().replace("svg:", "")))
    else:
        drawer = Draw.MolDraw2DCairo(w, h)  #cairo requires anaconda rdkit
        # opts = drawer.drawOptions()
        if not use_similarity_map:
            drawer.DrawMolecule(mol)
        else:
            SimilarityMaps.GetSimilarityMapFromWeights(mol,
                                                       weights,
                                                       draw2d=drawer)
        drawer.FinishDrawing()
        #
        # with open("/home/shuwang/sandbox/tmp.png","wb") as f:
        #     f.write(drawer.GetDrawingText())

        import io
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        buff = io.BytesIO()
        buff.write(drawer.GetDrawingText())
        buff.seek(0)
        plt.figure()
        i = mpimg.imread(buff)
        plt.imshow(i)
        plt.show()
        # display(SVG(drawer.GetDrawingText()))


def visualise_charges(mol,
                      show_hydrogens=False,
                      condense_hydrogen_charges=True,
                      property_name="PartialCharge",
                      drop_leading_zero=True,
                      **kwargs):
    if condense_hydrogen_charges and not show_hydrogens:
        mol = Chem.Mol(mol)
        for at in mol.GetAtoms():
            if at.GetAtomicNum() == 1 and at.GetDegree() == 1:
                nbr = at.GetBonds()[0].GetOtherAtom(at)
                if nbr.GetAtomicNum() != 1:
                    nbr.SetDoubleProp(
                        property_name,
                        nbr.GetDoubleProp(property_name) +
                        at.GetDoubleProp(property_name))
    if not show_hydrogens:
        mol = Chem.RemoveHs(mol)
    atom_mapping = {}
    for idx, atm in enumerate(mol.GetAtoms()):
        try:
            #currently only designed with partial charge in mind
            # keeps 2.s.f, and optional removes starting `0.``
            tmp = f"{atm.GetDoubleProp(property_name):.2g}"
            if drop_leading_zero:
                tmp = tmp.replace('0.', '.')
            if tmp[0] in ('0', '.'):
                tmp = '+' + tmp
            atom_mapping[idx] = tmp
        except Exception as e:
            print("Failed at atom number {} due to {}".format(idx, e))
            return
    _draw_mol_with_property(mol, atom_mapping, **kwargs)


def visualize_charges(mol,
                      show_hydrogens=False,
                      property_name="PartialCharge",
                      **kwargs):
    return visualise_charges(mol, show_hydrogens, property_name, **kwargs)
