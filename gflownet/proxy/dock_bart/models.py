import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import (
    PreNormEncoderLayer,
    PreNormDecoderLayer,
)

#*************************    BART ***********************************************

class BART(nn.Module):
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        activation,
        max_seq_len,
        drop_out,
    ):
        super().__init__()

        print(f'BART Module Init()')

#----------------------hyper parameters to be saved--------------------------------------------
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.drop_out = drop_out
        self.activation = activation
        self.max_seq_len = max_seq_len
#---------------------------------------------------------------------------------
        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, drop_out, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_norm = nn.LayerNorm(d_model)
        dec_layer = PreNormDecoderLayer(d_model, num_heads, d_feedforward, drop_out, activation)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(drop_out)
        self.register_buffer("pos_emb", self._positional_embs())

        '''
        #for log graphy only -- not working
        collate_output = {
            "encoder_input": torch.zeros(([60, 128])),
            "encoder_pad_mask": torch.zeros(([60, 128])),
            "decoder_input": torch.zeros(([62, 128])),
            "decoder_pad_mask": torch.zeros(([62, 128])),
            "target": torch.zeros(([62, 128])),
            "target_mask": torch.zeros(([62, 128])),
            "target_smiles": torch.zeros(128,)
        }

        dummy = [collate_output]
        self.example_input_array = collate_output
        '''

        self._init_params()

    def get_params(self):
        return {
            'vocab_size':self.vocab_size, 
            'd_model':self.d_model,
            'num_layers':self.num_layers, 
            'num_heads':self.num_heads,
            'd_feedforward':self.d_feedforward,
            'dropout':self.drop_out,
            'activation':self.activation,
            'max_seq_len':self.max_seq_len
        }

    def _positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz, device="cpu"):
        """ 
        Method copied from Pytorch nn.Transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode 
        """

        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        encoder_input = x["encoder_input"]
        decoder_input = x["decoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
        decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)
        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=encoder_embs.device).half()
        #print(f'forward encoder: encoder_embs = {encoder_embs.dtype}, encoder_pad_mask = {encoder_pad_mask.dtype}')
        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        memory_pad_mask= encoder_pad_mask.clone()
        #print(f'forward decoder: decoder_embs={decoder_embs.dtype}, memory={memory.dtype}, tgt_mask={tgt_mask.dtype},decoder_pad_mask={decoder_pad_mask.dtype},memory_pad_mask={memory_pad_mask.dtype}')
        model_output = self.decoder(
            decoder_embs,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask
        )

        token_output = self.token_fc(model_output)
        #print(f'forward token_fc: model_output={model_output.dtype}, token_output={token_output.dtype}')

        output = {
            "model_output": model_output,
            "token_output": token_output
        }

        return output

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def encode(self, batch):
        """ Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)

        #print(f'encoder: encoder_embs={encoder_embs.dtype},encoder_pad_mask={encoder_pad_mask.dtype}')

        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, batch):
        """ Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        memory_input = batch["memory_input"].half()
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_embs.device).half()

        #print(f'decode: decoder_embs={decoder_embs.dtype},memory_input={memory_input.dtype},decoder_pad_mask={decoder_pad_mask.dtype},memory_pad_mask={memory_pad_mask.dtype},tgt_mask={tgt_mask.dtype}')

        model_output = self.decoder(
            decoder_embs, 
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask
        )
        token_output = self.token_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_input = batch_input["encoder_input"]
        enc_mask = batch_input["encoder_pad_mask"]

        # Freezing the weights reduces the amount of memory leakage in the transformer
        #self.freeze()

        encode_input = {
            "encoder_input": enc_input,
            "encoder_pad_mask": enc_mask
        }
        #print(f'sample_molecules:enc_input={enc_input.dtype},enc_mask={enc_mask.dtype}')
        memory = self.encode(encode_input)
        mem_mask = enc_mask.clone()
        #print(f'sample_molecules encode: encode:memory={memory.dtype}, mem_mask={mem_mask.dtype}')

        _, batch_size, _ = tuple(memory.size())

        decode_fn = partial(self._decode_fn, memory=memory, mem_pad_mask=mem_mask)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size, memory.device)

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, memory.device, k=self.num_beams)

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        #self.unfreeze()

        return mol_strs, log_lhs


    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask
        }
        #print(f'_decode_fn: pad_mask:{pad_mask.dtype}, memory:{memory.dtype}, mem_pad_mask:{mem_pad_mask.dtype}')
        model_output = self.decode(decode_input)
        #print(f'_decode_fn decode: model_output:{model_output.dtype}')
        return model_output

#******************************************************************************

#above is BART


#*************************  Dock ***********************************************

class Dock(nn.Module):
    def __init__(
        self,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        activation,
        max_seq_len,
        drop_out,
    ):
        super().__init__()

        print(f'BART Module Init()')

#----------------------hyper parameters to be saved--------------------------------------------
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.drop_out = drop_out
        self.activation = activation
        self.max_seq_len = max_seq_len
#---------------------------------------------------------------------------------

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, drop_out, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        self.num_d = 5
        self.ln1 = nn.LayerNorm(d_model * (1 + 2 * self.num_d))
        self.fc1 = nn.Linear(d_model * (1 + 2 * self.num_d), 2 * d_feedforward)
        self.dropout1 = nn.Dropout(drop_out)
        
        self.ln2 = nn.LayerNorm(2 * d_feedforward)
        self.fc2 = nn.Linear(2 * d_feedforward, d_feedforward)
        self.dropout2 = nn.Dropout(drop_out)
        
        self.pred_ln = nn.LayerNorm(d_feedforward)
        self.pred_fc = nn.Linear(d_feedforward, 1)

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(drop_out)
        self.loss_fn = nn.MSELoss()
        
        self.register_buffer("pos_emb", self._positional_embs())


        self._init_params()
        self._load_from_bart = 0

    def get_params(self):
        return {
            'vocab_size':self.vocab_size, 
            'd_model':self.d_model,
            'num_layers':self.num_layers, 
            'num_heads':self.num_heads,
            'd_feedforward':self.d_feedforward,
            'dropout':self.drop_out,
            'activation':self.activation,
            'max_seq_len':self.max_seq_len
        }

    def _positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs


    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        encoder_input = x["encoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
        # if self._load_from_bart:
        #     with torch.no_grad():
        #         encoder_embs = self._construct_input(encoder_input)
        #         memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask) #(s, b, d)
        #         memory_pad_mask= encoder_pad_mask.clone() #(b, s)
        # else:
        encoder_embs = self._construct_input(encoder_input)

        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask) #(s, b, d)
        memory_pad_mask= encoder_pad_mask.clone() #(b, s)
        N = 1e5
        mask_p = memory_pad_mask.transpose(0, 1).unsqueeze(-1).tile(1, memory.shape[-1])
        x_avg = (~mask_p * memory).sum(0) / (~mask_p).sum(0)
        x_max = torch.sort(memory + mask_p * (-N), 0)[0][-self.num_d:]
        x_max = torch.cat(tuple(x_max), -1)
        x_min = torch.sort(memory + mask_p * N, 0)[0][:self.num_d]
        x_min = torch.cat(tuple(x_min), -1)
        x = torch.cat((x_avg, x_max, x_min), -1).to(memory.dtype)
        x = self.ln1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.ln2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.pred_ln(x)
        x = self.pred_fc(x)
        x = x.squeeze(-1)

        return x

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs
    
    def load_from_bart(self, bart: BART):
        self._load_from_bart = 1
        self.emb.load_state_dict(bart.emb.state_dict())
        # self.pos_emb = torch.clone(bart.pos_emb)
        self.encoder.load_state_dict(bart.encoder.state_dict())

    

#******************************************************************************