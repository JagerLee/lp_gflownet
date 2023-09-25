import os
import sys
from typing import Any
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from tokeniser import MolEncTokeniser
from pre_train import BARTModel
from .config import *
import torch.nn as nn
import torch
from SmilesPE.tokenizer import SPE_Tokenizer


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
    bart_model.load_state_dict(save_dict["state_dict"])
    bart_model.to(device)
    
    return bart_model, tokeniser

class Model(nn.Module):
    def __init__(
        self,
        bart_model: BARTModel,
        tokeniser: MolEncTokeniser,
        device,
        precision,
        readable
    ):
        super().__init__()
        self.bart_model = bart_model
        
        DEFAULT_tokeniser = MolEncTokeniser.from_vocab_file(
            vocab_path=DEFAULT_SMILES_VOCAB_PATH,
            regex=REGEX_SMILES,
            chem_tokens_start_idx=DEFAULT_SMILES_CHEM_TOKEN_START
        )
        self.tokeniser = tokeniser if tokeniser is not None else DEFAULT_tokeniser
        
        self.device = device
        self.precision = precision
        self.pad_token_idx = self.tokeniser.vocab[self.tokeniser.pad_token]
        self.begin_token_idx = self.tokeniser.vocab[self.tokeniser.begin_token]
        self.vocab_size = bart_model.vocab_size
        self.readable = readable
        
        self.out = nn.Linear(self.vocab_size, self.vocab_size)
        
        nn.init.eye_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)
        
        self.to(device=device)
    
    def bart(self, states) -> torch.Tensor:
        batch_size = len(states)
        
        # switch states to smiles
        smis = []
        for state in states:
            smi_list = []
            token_list = self.readable(state)
            for token in token_list[1:]:
                if token == self.tokeniser.end_token:
                    break
                smi_list.append(token)
            smi = ''.join(smi_list)
            smis.append(smi)
        
        # tokenise smiles to tokens ids
        token_output = self.tokeniser.tokenise(smis, pad=True, mask=False)
        token_ids = token_output['original_tokens']     # [[1,2,3,0], [1,2,3,4], [1,2,0,0]]
        pad_mask = token_output["original_pad_masks"]   # [[0,0,0,1], [0,0,0,0], [0,0,1,1]]
        token_ids = torch.tensor(token_ids, dtype=self.precision, device=self.device)
        pad_mask = torch.tensor(pad_mask, dtype=torch.bool, device=self.device)
        
        
        # SmilesPE tokenizer
        # with open('SPE_ChEMBL.txt', 'r') as f:
        #     spe = SPE_Tokenizer(f)
        # with open('smilespe_vocab.txt', 'r') as f:
        #     vocab = f.readlines()
        # vocab = [v.rstrip() for v in vocab]
        # vocab = dict(zip(vocab, list(range(len(vocab)))))
        # token_ids = []
        # for smi in smis:
        #     tokens = spe.tokenize(smi).split(' ')
        #     token_ids.append([vocab.get(token, '?') for token in tokens])
        # TODO
        # padding
        
        
        memory = torch.zeros(
            (1, batch_size, self.bart_model.d_model)
        ).to(self.device)
        mem_mask = torch.zeros(
            (1, batch_size), dtype=torch.bool
        ).to(self.device)
        pad_mask = (token_ids == self.pad_token_idx)
        logsftm = self.bart_model._decode_fn(
            token_ids=token_ids,
            pad_mask=pad_mask,
            mem_pad_mask=mem_mask,
            memory=memory
        )
        bart_out = logsftm[-1]
        
        return bart_out.to(dtype=self.precision, device=self.device)
    
    def forward(self, states) -> Any:
        bart_out = self.bart(states) # (batch, vocab)
        y = bart_out
        
        y = self.out(y)
        
        return y
    
    # def __call__(self, states) -> torch.Tensor:
    #     batch_size = len(states)
    #     for i in range(states.shape[1]):
    #         if torch.all(states[:, i] == self.pad_token_idx):
    #             break
    #     token_ids = states[:, :i].clone().transpose(0, 1).long()
    #     if i == 0:
    #         token_ids = torch.ones((1, batch_size)).long().to(self.device) * self.begin_token_idx
    #     for i in range(batch_size):
    #         if torch.all(token_ids[:, i] == self.pad_token_idx):
    #             token_ids[0, i] = self.begin_token_idx
    #     memory =  torch.zeros(
    #         (1, batch_size, self.bart_model.d_model)
    #     ).to(self.device)
    #     mem_mask = torch.zeros(
    #         (1, batch_size), dtype=torch.bool
    #     ).to(self.device)
    #     pad_mask = (token_ids == self.pad_token_idx)
    #     logits = self.bart_model._decode_fn(
    #         token_ids=token_ids,
    #         pad_mask=pad_mask,
    #         mem_pad_mask=mem_mask,
    #         memory=memory
    #     )
        
    #     return logits[-1].to(dtype=self.precision, device=self.device)
    
    # def parameters(self):
    #     return self.bart_model.token_fc.parameters()
    
    def train(self):
        self.bart_model.train()
        self.out.train()
    
    def eval(self):
        self.bart_model.eval()
        self.out.eval()
    
    def parameters(self):
        return [i for i in self.bart_model.token_fc.parameters()] + [j for j in self.out.parameters()]