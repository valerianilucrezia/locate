from locate.utils import *
import numpy as np

def all_stopping_criteria(old, new, e, step):
    """_summary_

    Parameters
    ----------
    old : _type_
        _description_
    new : _type_
        _description_
    e : _type_
        _description_
    step : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    old = collect_params(old)
    new = collect_params(new)
    diff_mix = np.abs(old - new) / np.abs(old)
    
    if np.all(diff_mix < e):
        return [True, diff_mix]
    return [False, diff_mix]