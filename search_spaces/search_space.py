from abc import ABC, abstractmethod
import numpy as np
from search_spaces.utils import snap_to_valid_anasod


class SearchSpace(ABC):

    _name = 'base'

    def __init__(self, file_path, dataset, device,):
        """Base class for model loader (load a trained ensemble/single neural architecture from disk"""
        self.filt_path = file_path
        self.api = dataset
        self.device = device
        self.available_seeds = None
        # this specifies the seeds on which models are trained. the length of this is the number of components of deep ensemble

        self.ops = None
        self.num_ops = None
        self.op_spots = None

    @abstractmethod
    def query(self, arch, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def to_networkx(self, arch, *args, **kwargs):
        """Convert an architecture string or index into a networkx format representation of the neural architecture
        cell."""
        raise NotImplementedError

    def get_random_anasod(self, alpha=None, normalized=True):
        # Get random anasod encoding from the search space via Dirichlet sampling
        if alpha is None:
            alpha = np.ones(self.num_ops)
        normalised_encoding = np.random.dirichlet(alpha=alpha)
        encoding = snap_to_valid_anasod(normalised_encoding, self.op_spots)
        if normalized:
            return encoding
        return self.op_spots * encoding
    #
    # @abstractmethod
    # def sample_from_anasod(self,  anasod_encoding, *args, **kwargs):
    #     """Given an anasod encoding, return a sample architecture parameterised by this encoding."""
    #     raise NotImplementedError

    @abstractmethod
    def get_random_arch(self, *args, **kwargs):
        pass

    # @abstractmethod
    # def mutate(self, arch_str, **kwargs):
    #     pass
