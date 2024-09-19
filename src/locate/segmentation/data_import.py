import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# specific function to call if input file is a .tsv
def get_data_tsv(data: Path, frequencies: list[str]) -> dict[str, np.ndarray]:
    """Extract allele frequency data for specified frequencies from a text file and return as a dictionary.

    Args:
        data (pathlib Path): Path holding input data file

        frequencies (list[str]): List of frequencies to be extracted and analyzed, should correspond to column names.

    Returns:
        (dict[str, np.ndarray]): Dictionary with key=str in frequency list and value=allele frequency associated with that column.

    Raises:
        (RuntimeError): If returned dict will be empty after attempting to find column names match those in supplied frequencies list.
    """
    # return dict to keep track of variable names
    ret_dict = {}
    snv = pd.read_csv(data, sep = '\t')
    snv.fillna(value=0, axis = 0, inplace = True)
    
    for i in frequencies:
        if i in snv:
            ret_dict[i] = np.array(snv[i])
        else:
            warnings.warn(message=f"{i} not found as column name in input data file. If this is unexpected, please check argument 'frequencies' for typos. Passed 'frequencies' list: {frequencies}")
    
    # check if dict is still empty. 'not ret_dict' on an empty dict returns True
    if not ret_dict:
        raise RuntimeError(f"No column names in input data file matched the names supplied in frequencies list.")
    
    return ret_dict

# specific funtion to call if a function is a .csv
# FIXME remove bps/cna_id after we know that sims work
def get_data_csv(data: Path, frequencies: list[str]) -> dict[str, np.ndarray]:
    """Extract allele frequency data for specified frequencies from a CSV file and return as a dictionary.

    Args:
        data (pathlib Path): Path holding input data file

        frequencies (list[str]): List of frequencies to be extracted and analyzed, should correspond to column names.

    Returns:
        (dict[str, np.ndarray]): Dictionary with key=str in frequency list and value=allele frequency associated with that column.

    Raises:
        (RuntimeError): If returned dict will be empty after attempting to find column names match those in supplied frequencies list.
    """
    ret_dict = {}

    snv = pd.read_csv(data)
    snv.fillna(value='normal', axis = 0, inplace = True)
    snv.sort_values(by = ['pos'], inplace = True)
    snv['id'] = [i for i in range(snv.shape[0])]
                    
    # bps_max = list(snv.groupby(['cna_id']).max(['pos']).id)
    # bps_min = list(snv.groupby(['cna_id']).min(['pos']).id)               
    # vaf_bps = snv.groupby('cna_id')

    for i in frequencies:
        if i in snv:
            ret_dict[i] = np.array(snv[i])
        else:
            warnings.warn(message=f"{i} not found as column name in input data file. If this is unexpected, please check argument 'frequencies' for typos. Passed 'frequencies' list: {frequencies}")
    
    # check if dict is still empty. 'not ret_dict' on an empty dict returns True
    if not ret_dict:
        raise RuntimeError(f"No column names in input data file matched the names supplied in frequencies list.")
    
    # ret_dict["bps"] = np.array(bps_max + bps_min)
    
    return ret_dict
