from pathlib import Path
import pandas as pd
import numpy as np
import ast


# function to load the tissue_dataset.csv file from parent directory
def tissue_dataset():
    """
    Loads the dataset of fibrosis textures.

    Returns:
        df: pandas DataFrame
    """

    def convert_to_numpy(x):
        return np.array(ast.literal_eval(x))

    path = Path(__file__).parent / "tissue_dataset.csv"
    df = pd.read_csv(path)
    df['Tissue Matrix'] = df['Tissue Matrix'].apply(convert_to_numpy)
    df['Tissue size'] = df['Tissue size'].apply(ast.literal_eval)
    return df
