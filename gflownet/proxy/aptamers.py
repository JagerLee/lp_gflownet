import numpy as np
import numpy.typing as npt

from gflownet.proxy.base import Proxy

import torch


class Aptamers(Proxy):
    """
    DNA Aptamer oracles
    """

    def __init__(self, oracle_id, norm, **kwargs):
        super().__init__(**kwargs)
        self.type = oracle_id
        self.norm = norm

    def setup(self, env=None):
        self.max_seq_length = env.max_seq_length

    def __call__(self, states: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        args:
            states : ndarray
        """
        def _length(x):
            if self.norm:
                return -1.0 * np.sum(x, axis=1) / self.max_seq_length
            else:
                return -1.0 * np.sum(x, axis=1)
        
        def _average(x):
            count = [0,0,0,0]
            for s in x:
                for i in range(4):
                    if s == i + 1:
                        count[i] += 1
            # return min(count) / sum(count) if sum(count) > 0 else 0
            r1 = 0
            if x[0] == 2:
                r1 += 0.5
            if x[1] == 2:
                r1 += 0.5
            r2 = sum(count[:2]) / sum(count) if sum(count) > 0 else 0
            return r1 * r2

        if self.type == "length":
            return torch.tensor(_length(states), device=self.device, dtype=self.float)
        elif self.type == "average":
            pro = []
            for s in states:
                pro.append(_average(s))
            return torch.tensor(pro, device=self.device, dtype=self.float)
        else:
            raise NotImplementedError("self.type must be length or average")
