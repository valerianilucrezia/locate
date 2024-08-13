from abc import ABC, abstractmethod
import torch

class Model(ABC):
    """
    General Structure for a Model
    """

    def __init__(self, data_dict, data_name):
        data = {k: v for k, v in data_dict.items() if k in data_name}
        self._data = data
        super().__init__()

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def guide(self):
        pass

    def set_params(self, params_dict):
        self._params.update(params_dict)