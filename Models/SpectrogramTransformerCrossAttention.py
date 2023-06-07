import torch
from torch import nn
from torch import functional as F
import math

from typing import Callable
from torch import Tensor

class AttentionPool(nn.Module):
    def __init__(self,
                 ni: int,
                 nheads: int,
                 bias: bool = True,
                 norm: Callable[[int], nn.Module] = nn.LayerNorm
    ):
        super().__init__()
        self.norm = norm(ni)
        self.vk = nn.Linear(ni, ni * 2, bias=bias)
        self.proj = nn.Linear(ni, ni)
        self.nheads = nheads
        self.head_width = int(ni / nheads)
        self.ni = ni

        self.o = nn.Linear(ni, ni, bias=bias)

        self.cls_q = nn.Parameter(torch.ones(self.nheads, self.head_width))

    def _init_weights(self):
        torch.nn.init.uniform_(self.cls_q, a=-1/self.head_width, b=1/self.head_width)

    def forward(self, x):
        x = self.norm(x.flatten(2))
        B, N, C = x.shape

        q = self.cls_q.expand(B, -1, -1)
        k, v = self.vk(x).reshape(B, N, 2, self.nheads, self.head_width).permute(2, 0, 3, 1, 4).chunk(2, 0)
        k = k[0, :, :, :, :]
        v = v[0, :, :, :, :]

        attn = (torch.unsqueeze(q, 2) @ k.transpose(-2, -1)) / math.sqrt(self.head_width)  # [B, nheads, 1, N]
        attn = nn.functional.softmax(attn, dim=-1)
        outputs = attn @ v
        outputs = torch.squeeze(outputs).reshape(B, self.ni)
        outputs = self.o(outputs)

        return outputs


# Borrowed and modified from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
       # >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=600, device="cpu"):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(500.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)

        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, n_fft, n_rri_inp, dropout=0.1, device="cpu", enable_rri=True, enable_spec=True, cut_spec=(2, 18)):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
                              'lower.') from e

        self.device = device
        self.ninp = ninp
        self.n_fft = n_fft

        self.enable_rri = enable_rri
        if enable_rri:
            self.rri_conv_net = nn.Sequential(
                nn.BatchNorm1d(1),
                torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                torch.nn.Conv1d(in_channels=32, out_channels=n_rri_inp, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(n_rri_inp),
            )

            self.rri_pos_encoder = PositionalEncoding(n_rri_inp, dropout, device=device)
            self.rri_cross_mha = torch.nn.MultiheadAttention(ninp, nhead, kdim=ninp, vdim=ninp)
            self.rri_layer_norm = nn.LayerNorm(ninp)
            self.rri_linear = nn.Sequential(
                nn.Linear(ninp, nhid),
                nn.ReLU(),
                nn.Linear(nhid, ninp)
            )
            self.rri_out_layer_norm = nn.LayerNorm(ninp)

        self.enable_spec = enable_spec
        if enable_spec:
            self.cut_spec = cut_spec
            self.pos_encoder = PositionalEncoding(ninp, dropout, device=device)
            encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

            self.stft_window = torch.hann_window(n_fft, device=device)

            if cut_spec is not None:
                print(cut_spec)
                self.stft_layer_norm = nn.LayerNorm(cut_spec[1] - cut_spec[0])
                self.stft_expand_layer = nn.Sequential(
                    nn.Linear(cut_spec[1]-cut_spec[0], ninp),
                    nn.ReLU(),
                    nn.LayerNorm(ninp))
            else:
                self.stft_layer_norm = nn.LayerNorm(n_fft // 2)
                self.stft_expand_layer = nn.Sequential(
                    nn.Linear(n_fft//2, ninp),
                    nn.ReLU(),
                    nn.LayerNorm(ninp))

            self.attention_pooling = AttentionPool(n_rri_inp, nheads=4) # lambda x: torch.mean(x, dim=-2)
            self.layer_norm = nn.LayerNorm(ninp)

        decoder_input_size = (ninp if enable_spec else 0)

        self.decoder1 = nn.Linear(decoder_input_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.decoder2 = nn.Linear(128, ntoken)

        self.init_weights()

    def fix_transformer_params(self, fix_spec=True, fix_rri=True):

        for param in self.rri_cross_mha.parameters():
            param.requires_grad = (not fix_rri)

        for param in self.rri_conv_net.parameters():
            param.requires_grad = (not fix_rri)

        for param in self.rri_linear.parameters():
            param.requires_grad = (not fix_rri)

        for param in self.rri_layer_norm.parameters():
            param.requires_grad = (not fix_rri)

        for param in self.rri_out_layer_norm.parameters():
            param.requires_grad = (not fix_rri)


        for param in self.stft_layer_norm.parameters():
            param.requires_grad = (not fix_spec)

        for param in self.stft_expand_layer.parameters():
            param.requires_grad = (not fix_spec)

        for param in self.transformer_encoder.parameters():
            param.requires_grad = (not fix_spec)

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder1.bias)
        nn.init.zeros_(self.decoder2.bias)
        nn.init.uniform_(self.decoder1.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder2.weight, -initrange, initrange)

    def stft_and_reshape(self, src):
        ecg_stfts = torch.stft(src, n_fft=self.n_fft, return_complex=True, window=self.stft_window)

        # Cut spectrogram
        if self.cut_spec is not None:
            src = torch.log(torch.abs(ecg_stfts[:, self.cut_spec[0]:self.cut_spec[1], :]) + 1e-9)
        else:
            src = torch.log(torch.abs(ecg_stfts[:, :-1, :]) + 1e-9)

        src = torch.permute(src, [2, 0, 1])

        src = self.stft_layer_norm(src)
        src = self.stft_expand_layer(src)

        return src

    def construct_padding_mask(self, batch_size, padded_len, rri_len):
        padding = torch.arange(padded_len, device=self.device).repeat([batch_size, 1])
        padding = (padding >= rri_len[:, None]).type(torch.bool)

        return padding

    def pad_queries_to_0(self, Q, rri_len):
        padding = torch.arange(Q.shape[0], device=self.device)
        padding = padding.repeat([Q.shape[1], Q.shape[2], 1]).permute([2, 0, 1])
        Q[padding >= rri_len[None, :, None]] = 0
        return Q


    def forward(self, src, rri, rri_len=None):

        rri = torch.unsqueeze(rri, dim=1)
        rri = self.rri_conv_net(rri)
        rri = torch.permute(rri, [2, 0, 1])
        rri = self.rri_pos_encoder(rri)

        src = self.stft_and_reshape(src)
        src = self.pos_encoder(src)

        if rri_len is not None:
            rri = self.pad_queries_to_0(rri, rri_len)

        # Some cross attention!
        src = self.rri_cross_mha(rri, src, src)[0]
        src = self.rri_layer_norm(src)
        src = self.rri_linear(src) + src
        src = self.rri_out_layer_norm(src)

        output = self.transformer_encoder(src)
        output = torch.transpose(output, 0, 1)
        output = self.attention_pooling(output)

        output = self.decoder1(output)
        output = nn.functional.relu(output)
        output = self.dropout1(output)

        output = self.decoder2(output)

        return output
