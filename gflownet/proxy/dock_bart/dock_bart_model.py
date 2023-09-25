import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from typing import List, Dict

import numpy as np
from policy.molbart.tokeniser import MolEncTokeniser
from rdkit.Chem import MolToSmiles
import torch
from models import Dock
import selfies as sf
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

vocab_path = '/opt/anaconda3/envs/lp_reinvent-env/lib/python3.7/site-packages/reinvent_models/molbart/selfies.txt'

class DockBARTModel:
    def __init__(self, save_dict, specific_parameters: Dict):
        """
        :type activity_model: scikit-learn type of model object
        :type model_type: can be "classification" or "regression"
        """
        params = save_dict['params']
        state_dict = save_dict['state_dict']
        self.device = specific_parameters.get('device', 'cpu')
        self._activity_model = Dock(**params).to(self.device)
        torch.cuda.empty_cache()
        self._activity_model.load_state_dict(state_dict)
        self._load_tokeniser()
        print(f'convert model to {self.device}')
        self._activity_model.eval()

    def predict(self, molecules: List, parameters: Dict) -> np.array:
        """
        Takes as input RDKit molecules and uses a pickled scikit-learn model to predict activities.
        :param molecules: This is a list of rdkit.Chem.Mol objects
        :param parameters: Those are descriptor-specific parameters.
        :return: numpy.array with activity predictions
        """
        return self.predict_from_mols(molecules)

    def predict_from_mols(self, molecules: List):
        if len(molecules) == 0:
            return np.empty([])
        if sum([mol is not None for mol in molecules]) == 0:
            return [0.0] * len(molecules)
        batch = self.mols2batch(molecules)
        activity = self._activity_model(batch).tolist()
        for idx in range(len(molecules)):
            if idx in self.invalid_idx or activity[idx] > 0:
                activity[idx] = 0.0
        return activity

    def _load_tokeniser(self):
        self.tokeniser = MolEncTokeniser.from_vocab_file(
            vocab_path=vocab_path,
            chem_tokens_start_idx=6,
            regex="\[[^\]]+]"
        )
        self.pad_token_idx = self.tokeniser.vocab[self.tokeniser.pad_token]
        print('tokeniser loaded')
    
    def mols2batch(self, molecules):
        sels = []
        self.invalid_idx = []
        for idx, mol in enumerate(molecules):
            if mol:
                smi = MolToSmiles(mol)
            else:
                smi = None
            try:
                sel = sf.encoder(smi)
                sels.append(sel)
            except Exception as e:
                sels.append('[C]')
                self.invalid_idx.append(idx)
                continue
        enc_token_output = self.tokeniser.tokenise(sels, pad=True)
        enc_tokens = enc_token_output["original_tokens"]
        enc_pad_mask = enc_token_output["original_pad_masks"]
        enc_tokens, enc_pad_mask = self._check_seq_len(enc_tokens, enc_pad_mask)

        enc_token_ids = self.tokeniser.convert_tokens_to_ids(enc_tokens)

        enc_token_ids = torch.tensor(enc_token_ids, device=self.device).transpose(0, 1)
        enc_pad_mask = torch.tensor(enc_pad_mask, dtype=torch.bool, device=self.device).transpose(0, 1)

        collate_output = {
            "encoder_input": enc_token_ids,
            "encoder_pad_mask": enc_pad_mask
        }
        return collate_output
    
    def _check_seq_len(self, tokens, mask):
        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self._activity_model.max_seq_len:
            print(f"WARNING -- Sequence length {seq_len} is larger than maximum sequence size")

            tokens_short = [ts[:self._activity_model.max_seq_len] for ts in tokens]
            mask_short = [ms[:self._activity_model.max_seq_len] for ms in mask]
            return tokens_short, mask_short
        elif seq_len <= 10:
            tokens_short = [ts + [self.pad_token_idx] * (10 - len(ts)) for ts in tokens]
            mask_short = [ms + [self.pad_token_idx] * (10 - len(ms)) for ms in mask]
            return tokens_short, mask_short
        return tokens, mask