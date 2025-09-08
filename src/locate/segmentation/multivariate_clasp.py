from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

from locate.segmentation.multivariate_segmentation import MultivariateClaSPSegmentation, take_first_cp, validate_first_cp, find_cp_iterative
from locate.segmentation.data_import import get_data_csv, get_data_tsv


class MultivariateClaSP:    
    """Class that takes in a csv of allele frequencies and degredation rates to return change points identifying mutated regions.

    Args:
        input (str): The path to a csv file holding gene frequencies (vaf, baf, dr, etc.).
        mode (str): Denotes the method for deriving a change point, either sum, max, or mult.
        out_dir (str): The path to store outputs.
        frequencies (list): Strings denoting the gene frequencies to be used. default is ["vaf", "baf", "dr"]
        n_segments (str or int): The number of segments to split the time series into. By default, the numbers of segments is inferred automatically by applying a change point validation test. Defaults to learn.
        n_estimators (int): The number of ClaSPs in the ensemble. Defaults to 10.
        window_size (str or int): The window size detection method or size of the sliding window used in the ClaSP algorithm. Valid implementations include: 'suss', 'fft', and 'acf'. Defaults to suss
        k_neighbours (int): The number of nearest neighbors to use in the ClaSP algorithm. Defaults to 3
        distance (str): The name of the distance function to be computed for determining the k-NNs. Available options are "znormed_euclidean_distance" and "euclidean_distance". Defaults to euclidean_distance.
        score (str): The name of the scoring metric to use in ClaSP. Available options are "roc_auc" and "f1". Defaults to roc_auc.
        early_stopping (bool, optional): Determines if ensembling is stopped, once a validated change point is found or the ClaSP models do not improve anymore. Defaults to True.
        validation (str, optional): The validation method to use for determining the significance of the change point. The available methods are "significance_test" and "score_threshold". Defaults to significance_test.
        threshold (float, optional): The threshold value to use for the validation test. If the validation method is "significance_test", this value represents the p-value threshold for rejecting the null hypothesis. If the validation method is "score_threshold", this value represents the threshold score for accepting the change point. Defaults to 1e-15.
        excl_radius (int, optional): The radius (in multiples of the window size) around each point in the time series to exclude when searching for change points. Defaults to 5.
        n_jobs (int, optional): Amount of threads used in the ClaSP computation. Defaults to 1.
        random_state (int, optional): Sets random seed for reproducibility. Defaults to 2357.

    Attributes:
        multivariate_clasp_objects (dict): Dictionary holding the ClaSP attributes for each analyzed allele frequency.
        name (str): Unique name of the time series.
        all_cps (set): All unique change points found across all allele frequencies.
    """
    
    def __init__(self, 
                 input_path: str, 
                 mode: str, 
                 out_dir: str, 
                 cna_id = None,
                 frequencies: list=["vaf", "baf", "dr"], 
                 n_segments: str|int="learn",
                 n_estimators: int=10, 
                 window_size: str|int="suss", 
                 k_neighbours: int=3, 
                 distance: str="euclidean_distance", 
                 score: str="roc_auc", 
                 early_stopping: bool=True,
                 validation: str="significance_test", 
                 threshold: float=1e-15, 
                 excl_radius: int=5, 
                 n_jobs: int=1, 
                 random_state: int=2357):

        # take dict of inherited args to pass to BinaryClaSPSegmentation. Delete params only used with this class
        self.kwargs = locals()
        for i in ['self', 'input_path', 'mode', 'out_dir', 'frequencies', 'cna_id']:
            del self.kwargs[i]
            
        # assign as usual
        self.frequencies = frequencies
        self.input = Path(input_path)
        self.mode = mode
        self.out_dir = out_dir
        self.window_size = window_size
        self.threshold = threshold
        self.validation = validation
        self.k_neighbors = k_neighbours
        self.cna_id = cna_id

        self.get_data = None

        self._check_args()

    def _check_args(self):
        """Checks non-optional arguments when initializing MultivariateClaSP. Optional argument checking handled by claspy package.

        Raises:
            TypeError: If any given argument is not of correct type.
        """
        # assign variable to attribute to be called later
        if self.input.is_file() is True:
            # access via suffix because it is a pathlib object
            # if bugs occur in future probably just use str(Path(self.input)).endswith()
            if self.input.suffix == ".csv":
                self.get_data = get_data_csv
            elif self.input.suffix == ".tsv":
                self.get_data = get_data_tsv
        else:
            raise TypeError(f"input file must be csv or tsv, not {str(self.input.suffix)}")
        
        # should be all that is needed as input is already checked
        if self.out_dir is None:
            self.out_dir = self.input.parent
        else:
            self.out_dir = Path(self.out_dir)
        if self.out_dir.exists() is False:
            self.out_dir.mkdir()

        if type(self.mode) != str or self.mode in ["max", "sum", "mult"] is False:
            raise TypeError(f"mode must be string of one of the following options: max, sum, mult")
        # if all(self.kwargs["frequencies"], str) is False:
        #     raise TypeError(f"If frequencies is specified, list items must be strings")


    def analyze_time_series(self):        
        """Function to find breakpoints of multiple time series by passing to claspy BinaryClaSPSegmentation. Creates the multivariate_clasp_objects and all_cps attributes.
        """
        
        self.name = f'{self.mode}_{self.window_size}_{self.threshold}'

        # returns dictionary to preserve variable names
        original_data = self.get_data(self.input, self.frequencies)
        
        # update to account for frequencies not present in input csv
        self.frequencies = original_data.keys()

        # sort bps
        # FIXME remove bps after we know sims work
        # self.bps = np.sort(original_data["bps"])[1:-1]

        self.all_cps = set()
        # Move try except block out of here and into simulations.py, need to know if an error is thrown here
        clasp_kwargs = self.kwargs.copy()
        for i in ['n_segments', 'threshold', 'validation', 'window_size']:
            del clasp_kwargs[i]

        self.multivariate_clasp_objects = {}
        for i in self.frequencies:
            ts_obj = MultivariateClaSPSegmentation(**self.kwargs)
            ts_obj.initialize(original_data[i])
            self.multivariate_clasp_objects[i] = ts_obj
        
        validate_first_cp(self.multivariate_clasp_objects, cp=take_first_cp(self.multivariate_clasp_objects, self.mode))
        self.all_cps = find_cp_iterative(self.multivariate_clasp_objects, self.mode)

    def plot_results(self, title=None, save=False, figsize=(10, 8), dpi=100):
        """
        Plot timeseries with inferred change-points and improved aesthetics.
        
        Parameters
        ----------
        title : None | str, optional
            Title of the plot, by default None
        save : bool, optional
            Save plot, by default False
        figsize : tuple, optional
            Figure size (width, height) in inches, by default (10, 8)
        dpi : int, optional
            Resolution for saved figure, by default 100
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        from matplotlib.lines import Line2D
        
        variables = list(self.multivariate_clasp_objects.keys())
        n_vars = len(variables)
        
        # Create figure with more appropriate sizing and spacing
        fig, axs = plt.subplots(nrows=n_vars, ncols=1, figsize=figsize, sharex=True)
        if n_vars == 1:
            axs = [axs]  # Make axs always a list for consistent indexing
        
        # Set up colors for CNA categories
        if self.cna_id is not None:
            categories = np.unique(self.cna_id)
            cmap = plt.cm.get_cmap('Set2', len(categories))
            colors = {category: cmap(i) for i, category in enumerate(categories)}
            color_values = [colors[label] for label in self.cna_id]
            
            # Create custom legend for CNA categories
            handles = [Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=colors[label], markersize=8, 
                            label=f'CNA: {label}') for label in colors]
        else:
            color_values = ['#1f77b4' for _ in range(len(self.multivariate_clasp_objects[variables[0]].time_series))]
        
        # Add breakpoint legend handle
        bp_handle = Line2D([0], [0], color='indianred', linestyle='dashed', linewidth=1.5, 
                        label='Change Point')
        
        # Plot each variable
        for i, var in enumerate(variables):
            profile = self.multivariate_clasp_objects[var].time_series
            bps = self.all_cps
            x_values = list(range(len(profile)))
            
            # Plot data points with improved styling
            axs[i].scatter(x=x_values, y=profile, c=color_values, s=4, alpha=0.7)
            
            # Add vertical lines for breakpoints with improved styling
            if bps:
                y_range = max(profile) - min(profile)
                y_min = min(profile) - 0.05 * y_range
                y_max = max(profile) + 0.05 * y_range
                for bp in bps:
                    axs[i].axvline(x=bp, ymin=0, ymax=1, color='indianred', 
                                linestyle='dashed', linewidth=1.5, alpha=0.8)
            
            # Set labels with better formatting
            axs[i].set_ylabel(f"{var.upper()}", fontweight='bold')
            
            # Set specific y-limits for certain variables
            if var.lower() in ['baf', 'vaf']:
                axs[i].set_ylim(0, 1)
                axs[i].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            
            # Add gridlines for better readability
            axs[i].grid(True, linestyle='--', alpha=0.3)
            
            # Remove top and right spines for cleaner look
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
        
        # Set common x-label on bottom axis only
        axs[-1].set_xlabel('Position', fontweight='bold')
        
        # Add title with better formatting
        if title is not None:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        # Combine legend items and place in a better location
        all_handles = ([bp_handle] + handles) if self.cna_id is not None else [bp_handle]
        fig.legend(handles=all_handles, loc='upper center', 
                bbox_to_anchor=(0.5, 0.02), ncol=min(len(all_handles), 4), 
                frameon=True, fancybox=True, shadow=True)
        
        # Improve layout and spacing
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save figure with improved quality
        if save:
            fig.savefig(self.out_dir / f'{self.name}_timeseries.png', dpi=dpi, bbox_inches='tight')
            
        return fig, axs

    # def plot_results(self,
    #                 title: None | str = None,
    #                 save: bool = False):
        
    #     """Plot timeseries with inferred change-points

    #     Parameters
    #     ----------
    #     title : None | str, optional
    #         Title of the plot, by default None
    #     save : bool, optional
    #         Save plot, by default False
    #     """      
          
    #     variables = list(self.multivariate_clasp_objects.keys())
    #     fig, axs = plt.subplots(nrows = len(variables), ncols = 1, figsize=(8, 7))    
        
    #     if self.cna_id is not None:
    #         categories = np.unique(self.cna_id)
    #         colors = {category: matplotlib.colormaps['Set2'].colors[i] for i, category in enumerate(categories)}
    #         color_values = [colors[label] for label in self.cna_id]
    #     else:
    #         color_values = ['tab:blue' for i in range(len(self.multivariate_clasp_objects[variables[0]].time_series))]
        

    #     for i in range(0, len(variables)):
    #         profile = self.multivariate_clasp_objects[variables[i]].time_series
    #         bps = self.all_cps
    #         axs[i].scatter(x = [i for i in range(len(profile))], y = profile, c = color_values, s = 2)
    #         axs[i].set_ylabel(str(variables[i]))
    #         axs[i].set_xlabel('pos')
    #         axs[i].vlines(bps, ymin = min(profile) - 0.05, ymax = max(profile) + 0.05, colors = 'indianred', label = 'Predicted BP', linestyles = 'dashed')
            
    #         if variables[i] == 'baf' or variables[i] == 'vaf':
    #             axs[i].set_ylim(0,1)

    #     if title is not None:
    #         fig.suptitle(title)
        
    #     # Create proxy artists for legend
    #     if self.cna_id is not None:
    #         handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=label) for label in colors]
    #         fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1), ncol=len(categories), frameon=True)
            
    #     fig.tight_layout()
    #     if save:
    #         fig.savefig(self.out_dir / f'{self.name}_timeseries.png', dpi = 500)


    # def plot_combined_profile(self,
    #                           title: None | str = None, 
    #                           save: bool = False):
    #     """Plot CLASP profiles with identified change points

    #     Args:
    #         title (None | str, optional): A title for the produced figure. Defaults to None.
    #         save (bool, optional): Parameter specifying whether the figure should be saved to the out_dir. Defaults to False.
    #     """

        
    #     variables = list(self.multivariate_clasp_objects.keys())
    #     fig, axs = plt.subplots(nrows = len(variables), ncols = 1, figsize=(10, 8))

    #     for i in range(0, len(variables)):
    #         profile = self.multivariate_clasp_objects[variables[i]].profile
    #         bps = self.multivariate_clasp_objects[variables[i]].change_points
    #         axs[i].plot(profile, label=str(variables[i]))
    #         axs[i].set_ylabel(str(variables[i]))
    #         axs[i].vlines(bps, ymin = min(profile) - 0.05, ymax = max(profile) + 0.05, colors = 'tab:green', label = 'Predicted BP', linestyles = 'dashed')

    #     if title is not None:
    #         fig.suptitle(title)
            
    #     fig.tight_layout()

    #     if save:
    #         fig.savefig(self.out_dir / f'{self.name}_profiles.png', dpi = 500)
