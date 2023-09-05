import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from molbart.tokeniser import MolEncTokeniser
from molbart.pre_train import BARTModel
from molbart.config import *
import torch

def load_model(file_path, device):
    default_tokeniser = MolEncTokeniser.from_vocab_file(
                vocab_path=DEFAULT_SMILES_VOCAB_PATH,
                regex=REGEX_SMILES,
                chem_tokens_start_idx=DEFAULT_SMILES_CHEM_TOKEN_START
            )
    save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)
    network_params = save_dict.get("network_params", {})
    max_sequence_length = network_params.get('max_seq_len', 256)
    tokeniser:MolEncTokeniser = save_dict.get('tokeniser', default_tokeniser)
    network_params['pad_token_idx'] = tokeniser.vocab[tokeniser.pad_token]
    try:
        network_params['drop_out'] = network_params.pop('dropout')
    except:
        pass
    bart_model = BARTModel(**network_params)
    bart_model.to(device)
    
    return bart_model, tokeniser
