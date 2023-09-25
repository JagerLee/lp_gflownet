import numpy as np
import numpy.typing as npt

from gflownet.proxy.base import Proxy
from rdkit.Chem import MolFromSmiles
import torch
from rdkit import RDLogger
import os
import selfies as sf

from gflownet.proxy.dock_bart.dock_bart_model import DockBARTModel

RDLogger.DisableLog('rdApp.*')


class DockBART(Proxy):
    """
    DNA Aptamer oracles
    """

    def __init__(self, oracle_id, vocab_path, chem_begin_idx, model_path, **kwargs):
        super().__init__(**kwargs)
        self.type = oracle_id
        self.vocab_path = vocab_path
        self.chem_begin_idx = chem_begin_idx
        self.eos_token_idx = 3
        self.pad_token_idx = 0
        self.begin_token_idx = 2
        # root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        root_path = ''
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
        
        save_dict = torch.load(model_path)
        print(f'DockBART model loaded from {model_path}')
        self.model = DockBARTModel(
            save_dict=save_dict,
            specific_parameters=kwargs
        )

    def setup(self, env=None):
        self.max_seq_length = env.max_seq_length

    def __call__(self, states: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        args:
            states : ndarray
        """

        if self.type == "dockbart":
            mols = []
            alphabet = dict(zip(list(range(self.n_alphabet)), self.alphabet))
            for x in states:
                smi_list = []
                for s in x[1:]:
                    if s == self.eos_token_idx or s == self.pad_token_idx:
                        break
                    smi_list.append(alphabet[s])
                smi = ''.join(smi_list)
                try:
                    smi = sf.decoder(smi)
                    mol = MolFromSmiles(smi)
                except:
                    mol = None
                mols.append(mol)
            pro = self.model.predict_from_mols(mols)

            return -torch.tensor(pro, device=self.device, dtype=self.float)
        else:
            raise NotImplementedError("self.type must be dockbart")
