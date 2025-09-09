from locate.segmentation.claspy.utils import check_input_time_series, check_excl_radius
from locate.segmentation.claspy.clasp import ClaSPEnsemble, ClaSP
from locate.segmentation.claspy.validation import map_validation_tests, significance_test
import numpy as np

"""
Note: Some functions are rewritten from base Claspy due to being unable to reference variables and methods needed for computing a multivariate
Clasp profile in an object oriented manner.

Class: multivariateClaSPSegmentation
Class that takes in a time series of an allele frequency and finds change points. Inherits from BinaryClaSPSegmentation from claspy package
    Inputs:
        - input: allele frequency time series
        - mode: str of either sum, max, or multi. Denotes the method for deriving a change point
        - out_dir: str of path to store outputs
        - frequencies: list of strings denoting the gene frequencies to be used. default is ["vaf", "baf", "dr"]
        - n_segments: method for detecting time series segments. default is "learn"
        - n_estimators: default 10
        - window_size: default 5
        - k_neighbors: default 3
        - distance: default "euclidian_distance"
        - score: default "roc_auc"
        - early_stopping: default True
        - validation: method for detecting the significance of a change point. default is "significance_test"
        - threshold: float of significance threshold. default is 1e-15
        - excl_radius: default 5
        - n_jobs: default 1
        - random_state: default 2357
    Methods:
    get_first_cp: find first change point in a given time series
"""

class MultivariateClaSPSegmentation(ClaSP):
    """_summary_

    Parameters
    ----------
    ClaSP : _type_
        _description_
    """    
    
    def __init__(self, n_segments="learn",
                 n_estimators: int=10, 
                 window_size=10, 
                 k_neighbours: int=3, 
                 distance: str="euclidean_distance", 
                 score: str="roc_auc", 
                 early_stopping: bool=True,
                 validation: str="significance_test", 
                 threshold: float=1e-15, 
                 excl_radius: int=5, 
                 n_jobs: int=1, 
                 random_state: int=2357):
        
        super().__init__(window_size, k_neighbours, distance, score, excl_radius, n_jobs)

        self.min_seg_size = self.window_size * self.excl_radius
        self.threshold = threshold
        self.validation = validation
        self.n_estimators = n_estimators
        self.n_segments = n_segments
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.score = score
        self.distance = distance
        self.n_jobs = n_jobs
        self.k_neighbours = k_neighbours
        self.window_size = window_size
        self.excl_radius = excl_radius

    def initialize(self, time_series):
        """_summary_

        Parameters
        ----------
        time_series : _type_
            _description_
        """        
        
        self.time_series = time_series
        check_input_time_series(time_series)
        check_excl_radius(self.k_neighbours, self.excl_radius)

        self.n_timepoints = time_series.shape[0]
        self.min_seg_size = self.window_size * self.excl_radius
        
        self.queue = []
        self.clasp_tree = []
        
        if self.n_segments == "learn":
            self.n_segments = time_series.shape[0] // self.min_seg_size
            
        self.prange = 0, time_series.shape[0]    
        self.clasp = ClaSPEnsemble(n_estimators=self.n_estimators,
                        window_size=self.window_size,
                        k_neighbours=self.k_neighbours,
                        distance=self.distance,
                        score=self.score,
                        early_stopping=self.early_stopping,
                        excl_radius=self.excl_radius,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state).fit(time_series, 
                                                    validation=self.validation, 
                                                    threshold=self.threshold)
                
        self.cp = self.clasp.split(validation=self.validation, threshold=self.threshold)
        # print('first cp',cp)
        # # check if it is valid with significant test
        
        self.profile = self.clasp.profile

    
    def local_segmentation(self, lbound, ubound, change_points):
        """_summary_

        Parameters
        ----------
        lbound : _type_
            _description_
        ubound : _type_
            _description_
        change_points : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """        

        if ubound - lbound < 2 * self.min_seg_size: 
            return self.clasp_tree, self.queue, 0

        self.clasp = ClaSPEnsemble(
            n_estimators=self.n_estimators,
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            distance=self.distance,
            score=self.score,
            early_stopping=self.early_stopping,
            excl_radius=self.excl_radius,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        ).fit(self.time_series[lbound:ubound], validation=self.validation, threshold=self.threshold)

        cp = split(self.clasp, validation=self.validation, threshold=self.threshold)
        if cp is None: 
            return self.clasp_tree, self.queue, 0

        # FIXME: score is originally a string argument and here (and below) it is reclassified as int. Change name?
        tmp_score = self.profile[cp]

        if not cp_is_valid(lbound + cp, change_points, self.n_timepoints, self.min_seg_size):  #candidate, change_points, n_timepoints, min_seg_size
            tmp_score = 0

        self.clasp_tree.append(((lbound, ubound), self.clasp))   
        self.queue.append((-tmp_score, len(self.clasp_tree) - 1))
        return tmp_score

def split(clasp, sparse=True, validation="significance_test", threshold=1e-15):
    """
    Split the input time series into two segments using the change point location.

    Parameters
    ----------
    sparse : bool, optional
        If True, returns only the index of the change point. If False, returns the two segments
        separated by the change point. Default is True.
    validation : str, optional
        The validation method to use for determining the significance of the change point.
        The available methods are "significance_test" and "score_threshold". Default is
        "significance_test".
    threshold : float, optional
        The threshold value to use for the validation test. If the validation method is
        "significance_test", this value represents the p-value threshold for rejecting the
        null hypothesis. If the validation method is "score_threshold", this value represents
        the threshold score for accepting the change point. Default is 1e-15.

    Returns
    -------
    int or tuple
        If `sparse` is True, returns the index of the change point. If False, returns a tuple
        of the two time series segments separated by the change point.

    Raises
    ------
    ValueError
        If the `validation` parameter is not one of the available methods.
    """
    clasp._check_is_fitted()
    cp = np.argmax(clasp.profile)

    if validation is not None:
        validation_test = map_validation_tests(validation)
        if not validation_test(clasp, cp, threshold): return None

    if sparse is True:
        return cp

    return clasp.time_series[:cp], clasp.time_series[cp:]
    

def cp_is_valid(candidate, 
                change_points, 
                n_timepoints, 
                min_seg_size):
    """Function to check that a given change point is valid
    Input:
    - candidate: change point to check
    - change_points: list of change points
    - n_timepoints: length of time series??
    - min_seg_size: window_size * excl_radius
    Return:
        - Bool of whether the change point is valid or not"""
    
    for change_point in [0] + change_points + [n_timepoints]:
        left_begin = max(0, change_point - min_seg_size)
        right_end = min(n_timepoints, change_point + min_seg_size)
        
        if candidate in range(left_begin, right_end): 
            return False

    return True


def take_first_cp(multivariate_clasp_objects: dict, 
                  mode):
    """_summary_

    Parameters
    ----------
    multivariate_clasp_objects : dict
        _description_
    mode : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    
    profiles = [i.profile for i in multivariate_clasp_objects.values()]
    cps = [np.argmax(i) for i in profiles]
    scores = [max(i) for i in profiles]

    if mode == 'max':
        idx = np.argmax(scores)
        most_common_cp = cps[idx]
            
    elif mode == 'mult':
        most_common_cp = np.argmax(np.prod(np.array(profiles), axis=0))
        
    elif mode == 'sum':
        most_common_cp = np.argmax(np.sum(np.array(profiles), axis=0)/len(multivariate_clasp_objects.values()))
        
    return most_common_cp


# What is validation test referring to?
def validate_first_cp(multivariate_clasp_objects: dict,
                      cp):
    """_summary_

    Parameters
    ----------
    multivariate_clasp_objects : dict
        _description_
    cp : _type_
        _description_
    """    
    
    # validate clasp object for each one given as input
    for ts_obj in multivariate_clasp_objects.values():
        validation_test = map_validation_tests(ts_obj.validation)
        ts_obj.val = validation_test(ts_obj.clasp, cp, ts_obj.threshold)
    # append clasp object variables
    if any([i.val for i in multivariate_clasp_objects.values()]):
        for ts_obj in multivariate_clasp_objects.values():
            ts_obj.clasp_tree.append((ts_obj.prange, ts_obj.clasp))
            ts_obj.queue.append((-ts_obj.profile[cp], len(ts_obj.clasp_tree) - 1))
    

def find_cp_iterative(multivariate_clasp_objects, mode):
    """Iteratively find changepoints shared between segments.

    Args:
        multivariate_clasp_objects (dict): Dictionary of initialized multivariateClaSPSegmentation objects for each time series.
        mode (str): Method used to combine clasp scores and profiles.

    Returns:
        list: A sorted list of all unique change points shared between time series.
    """

    CP = []
    for ts_obj in multivariate_clasp_objects.values():
        ts_obj.scores = []

    # n_segments should be the same for each from what I can tell, so just get one
    n_segments = np.max([ts_obj.n_segments for ts_obj in multivariate_clasp_objects.values()])
    
    for _ in range(n_segments - 1):
        # for i in [len(i.queue) == 0 for i in multivariate_clasp_objects.values()]:
        #     print(i)
        if all([len(i.queue) == 0 for i in multivariate_clasp_objects.values()]):
            #print('Stop because queue is empty')
            break
        for ts_obj in multivariate_clasp_objects.values():
            if len(ts_obj.queue) > 0:
                ts_obj.priority, ts_obj.clasp_tree_idx = ts_obj.queue.pop() #dr_queue.get()
                (ts_obj.lbound, ts_obj.ubound), ts_obj.clasp = ts_obj.clasp_tree[ts_obj.clasp_tree_idx]
                # FIXME: Here if mysplit returns None an error would occur. Possible solution below. Alternative: try except
                split_ret = ts_obj.clasp.split(validation=ts_obj.validation, threshold=ts_obj.threshold)
                if split_ret is None:
                    split_ret = 0
                ts_obj.cp = ts_obj.lbound + split_ret
                ts_obj.profile[ts_obj.lbound:ts_obj.ubound - ts_obj.window_size + 1] = np.max([ts_obj.profile[ts_obj.lbound:ts_obj.ubound - ts_obj.window_size + 1], ts_obj.clasp.profile], axis=0)
                ts_obj.prof = ts_obj.clasp.profile
            
            else:
                ts_obj.cp = 0
                ts_obj.priority = 0
                ts_obj.prof = np.array([])
            
        # all_bound = [(i.lbound, i.ubound) for i in multivariate_clasp_objects.values()]
        

        all_cp = [i.cp for i in multivariate_clasp_objects.values()]
        all_score = [i.priority for i in multivariate_clasp_objects.values()]
        
        # fill with -inf to end of array for arrays smaller than max array size
        all_profile = [i.prof for i in multivariate_clasp_objects.values() if i.prof.shape[0] > 0]
        max_profile_shape = np.max([i.shape[0] for i in all_profile])
        for i in range(0, len(all_profile)):
            if all_profile[i].shape[0] < max_profile_shape:
                new = np.append(all_profile[i], np.full(shape=(max_profile_shape-all_profile[i].shape[0]), fill_value=-np.inf))
                all_profile[i] = new
        
        all_profile = np.array(all_profile)
        if mode == 'max': # max score
            keep_cp = all_cp[np.argmax(np.abs(all_score))]
            
        elif mode == 'mult':
            new_score = np.prod(all_profile, axis = 0) #prof_dr * prof_baf * prof_vaf
            # TODO: Need test case
            new_score[new_score == np.inf] = -10000
            test_cp = np.argmax(new_score)
            keep_cp = multivariate_clasp_objects["dr"].lbound + np.argmax(new_score)
            
            
        elif mode == 'sum':
            new_score = np.sum(all_profile, axis = 0)
            new_score[new_score == np.inf] = -10000

            n_series = len(multivariate_clasp_objects.keys())
            test_cp = np.argmax(new_score/n_series)
            keep_cp = multivariate_clasp_objects["dr"].lbound + np.argmax(new_score/n_series)
            
        
        if mode != 'max':
            all_val = [] 
            for ts_obj in multivariate_clasp_objects.values():
                if ts_obj.prof.size > 0:
                    val = significance_test(ts_obj.clasp, test_cp, ts_obj.threshold)
                    all_val.append(val)
                if (True in all_val):
                    CP.append(keep_cp)
        
        else:
            CP.append(keep_cp)

        for ts_obj in multivariate_clasp_objects.values():
            ts_obj.scores.append(-ts_obj.priority)
        
        # define new rrange
        lrange, rrange = (multivariate_clasp_objects["dr"].lbound, keep_cp), (keep_cp, multivariate_clasp_objects["dr"].ubound) 
        for prange in (lrange, rrange):
            
            low = prange[0]
            high = prange[1]

            for ts_obj in multivariate_clasp_objects.values():
                scores_tmp = ts_obj.local_segmentation(low, high, CP)
                if scores_tmp != None:
                    ts_obj.scores.append(scores_tmp)
    
    return sorted(list(set(CP)))
