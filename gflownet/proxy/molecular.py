import numpy as np
import numpy.typing as npt

from gflownet.proxy.base import Proxy
from rdkit.Chem import MolFromSmiles
import torch
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


class Molecular(Proxy):
    """
    DNA Aptamer oracles
    """

    def __init__(self, oracle_id, norm, vocab_path, chem_begin_idx, **kwargs):
        super().__init__(**kwargs)
        self.type = oracle_id
        self.norm = norm
        self.vocab_path = vocab_path
        self.chem_begin_idx = chem_begin_idx
        self.eos_token_idx = 4
        self.pad_token_idx = 0
        with open(vocab_path, 'r') as f:
            alphabet = f.readlines()
        self.alphabet = [s.rstrip() for s in alphabet]
        self.n_alphabet = len(self.alphabet)

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
        
        def smi2mol(x):
            alphabet = dict(zip(list(range(self.n_alphabet)), self.alphabet))
            smi_list = []
            for s in x:
                if s == self.eos_token_idx:
                    break
                elif s >= self.chem_begin_idx:
                    smi_list.append(alphabet[s - 1])
            smi = ''.join(smi_list)
            p = 0 if MolFromSmiles(smi) is None else 1
            return p

        if self.type == "length":
            return torch.tensor(_length(states), device=self.device, dtype=self.float)
        elif self.type == "smi2mol":
            pro = []
            for s in states:
                pro.append(smi2mol(s))
            return torch.tensor(pro, device=self.device, dtype=self.float)
        else:
            raise NotImplementedError("self.type must be length or average")
