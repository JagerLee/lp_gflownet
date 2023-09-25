import numpy as np
import numpy.typing as npt

from gflownet.proxy.base import Proxy
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import qed
import torch
from rdkit import RDLogger
import os

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
        self.eos_token_idx = 3
        self.pad_token_idx = 0
        self.begin_token_idx = 2
        root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(root_path, vocab_path), 'r') as f:
            alphabet = f.readlines()
        self.alphabet = [s.rstrip() for s in alphabet]
        self.n_alphabet = len(self.alphabet)
        self.token_brackets = {}
        for idx, token in enumerate(self.alphabet):
            num = 0
            for s in token:
                if s == '(':
                    num += 1
                elif s == ')':
                    num -= 1
            self.token_brackets[idx] = num

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
            for i, s in enumerate(x[1:]):
                if s == self.eos_token_idx or s == self.pad_token_idx:
                    if i + 2 <= len(x) - 1 and x[i + 2] >= self.chem_begin_idx:
                        return 0
                    break
                smi_list.append(alphabet[s])
            smi = ''.join(smi_list)
            p = 0 if MolFromSmiles(smi) is None else 1
            return p
        def _qed(x):
            alphabet = dict(zip(list(range(self.n_alphabet)), self.alphabet))
            smi_list = []
            for i, s in enumerate(x[1:]):
                if s == self.eos_token_idx or s == self.pad_token_idx:
                    if i + 2 <= len(x) - 1 and x[i + 2] >= self.chem_begin_idx:
                        return 0
                    break
                smi_list.append(alphabet[s])
            smi = ''.join(smi_list)
            mol = MolFromSmiles(smi)
            try:
                return 0 if mol is None else qed(mol)
            except:
                return 0

        if self.type == "length":
            return torch.tensor(_length(states), device=self.device, dtype=self.float)
        elif self.type == "smi2mol":
            pro = []
            for s in states:
                pro.append(smi2mol(s))
            return torch.tensor(pro, device=self.device, dtype=self.float)
        elif self.type == 'qed':
            pro = []
            for s in states:
                pro.append(_qed(s))
            return torch.tensor(pro, device=self.device, dtype=self.float)
        else:
            raise NotImplementedError("self.type must be length or average")
