from abc import ABC, abstractmethod
import torch

class Model(ABC):
    """_summary_

    Parameters
    ----------
    ABC : _type_
        _description_
    """

    def __init__(self, data_dict, data_name):
        # Get CUDA setting from data_dict params if available
        self._CUDA = data_dict.get('params', {}).get('CUDA', False)
        self.device = torch.device('cuda' if self._CUDA else 'cpu')
        
        # Move data to correct device
        data = {k: v for k, v in data_dict.items() if k in data_name}
        self._data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in data.items()}
        super().__init__()

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def guide(self):
        pass
    
    def set_params(self, params_dict):
        """Update model parameters.
        
        Args:
            params_dict: Dictionary of parameters to update
        """
        # Ensure _params exists
        if not hasattr(self, '_params'):
            self._params = {}
            
        # Update parameters
        self._params.update(params_dict)
        
        # Move any tensor parameters to correct device
        self._params = {
            k: v.to(self.device) if torch.is_tensor(v) else v 
            for k, v in self._params.items()
        }