import pandas as pd
import numpy as np


def convert_to_string(x):
    """
    Fonction requise pour désérialiser preprocesseur.joblib.
    Cette fonction doit exister au niveau module pour que joblib
    puisse la retrouver lors du chargement.
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.astype(str)
    if isinstance(x, np.ndarray):
        return x.astype(str)
    return x
