import torch
import torch.nn as nn
import math

class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention block
        att = self.norm1(src)
        att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        att = src + self.dropout1(att)

        # Feedforward block
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout2(out)
        return out

class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ):
        # Self attention block 
        query = self.norm1(tgt)
        query = self.self_attn(query, query, query, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        query = tgt + self.dropout1(query)

        # Context attention block
        att = self.norm2(query)
        att = self.multihead_attn(att, memory, memory, attn_mask=memory_mask, 
                key_padding_mask=memory_key_padding_mask)[0]
        att = query + self.dropout2(att)

        # Feedforward block
        out = self.norm3(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout3(out)
        return out

class _AbsTransformerModel(nn.Module):
    def __init__(
        self,
        pad_token_idx,
        vocab_size, 
        d_model,
        max_seq_len,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.register_buffer("pos_emb", self._positional_embs())

    def forward(self, x):
        raise NotImplementedError()

    

class BARTModel2(_AbsTransformerModel):
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
        dropout=0.1,
    ):
        super().__init__(
            pad_token_idx,
            vocab_size, 
            d_model,
            max_seq_len,
            dropout,
        )
        self.pad_token_idx = pad_token_idx
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.activation = activation
        self.max_seq_len = max_seq_len
        self.dropout_size = dropout
        
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.dropout_layer = nn.Dropout(dropout)

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_norm = nn.LayerNorm(d_model)
        dec_layer = PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

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
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=encoder_embs.device)

        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        model_output = self.decoder(
            decoder_embs,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=encoder_pad_mask.clone()
        )

        token_output = self.token_fc(model_output)

        output = {
            "model_output": model_output,
            "token_output": token_output
        }

        return output

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
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_embs.device)

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
    
    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask
        }
        model_output = self.decode(decode_input)
        return model_output

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            'pad_token_idx': self.pad_token_idx,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'd_feedforward': self.d_feedforward,
            'activation': self.activation,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout_size
        }
    
    def freeze(self) -> None:
        r"""
        Freeze all params for inference.

        Example::

            model = MyLightningModule(...)
            model.freeze()

        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters for training.

        .. code-block:: python

            model = MyLightningModule(...)
            model.unfreeze()

        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout_layer(embs)
        return embs

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


class BARTModel(nn.Module):
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
        schedule="cycle",
        warm_up_steps=None,
        drop_out=0.1,
        train_num_batches = 0,
        val_num_batches = 0,
        decode_sampler=None,
        lr=None,
        weight_decay=None,
        total_steps=None,
    ):
        super().__init__()

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
        self.train_num_batches = train_num_batches
        self.val_num_batches = val_num_batches

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
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=encoder_embs.device)
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

    def encode(self, batch) -> torch.Tensor:
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

    def decode(self, batch) -> torch.Tensor:
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
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_embs.device)

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
