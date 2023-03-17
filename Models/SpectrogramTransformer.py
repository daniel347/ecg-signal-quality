import torch
from torch import nn
from torch import functional as F
import math

from typing import Callable
from torch import Tensor

class AttentionPool2d(nn.Module):
    "Attention for Learned Aggregation"
    def __init__(self,
        ni:int,
        nheads:int,
        ncls:int,
        bias:bool=True,
        norm:Callable[[int], nn.Module]=nn.LayerNorm
    ):
        super().__init__()
        self.norm = norm(ni)
        self.q = nn.Linear(ni, ni, bias=bias)
        self.vk = nn.Linear(ni, ni*2, bias=bias)
        self.proj = nn.Linear(ni, ni)
        self.nheads = nheads
        self.head_width = int(ni/nheads)
        self.ncls = ncls

    def forward(self, x:Tensor, cls_q:Tensor):
        x = self.norm(x.flatten(2))
        B, N, C = x.shape

        q = self.q(cls_q.expand(B, N, -1, -1)).reshape(B, self.nheads, self.ncls, N, self.head_width)
        k, v = self.vk(x).reshape(B, N, 2, self.nheads, self.head_width).permute(2, 0, 3, 1, 4).chunk(2, 0)
        k = k[0, :, :, :, :].unsqueeze(2).repeat(1, 1, self.ncls, 1, 1)
        v = v[0, :, :, :, :].unsqueeze(2).repeat(1, 1, self.ncls, 1, 1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        attn = nn.functional.softmax(attn, dim=-1)
        head_outputs = torch.sum((attn @ v), dim=-2)
        head_outputs = head_outputs.transpose(-2, -3)
        head_outputs = head_outputs.reshape(B, self.ncls, C)

        return head_outputs, attn


class LearnedAggregation(nn.Module):
    "Learned Aggregation from https://arxiv.org/abs/2112.13692"
    def __init__(self,
        ni:int,
        attn_bias:bool=True,
        ffn_expand:int|float=3,
        ncls:int=1,
        nheads:int=1,
        norm:Callable[[int], nn.Module]=nn.LayerNorm,
        act_cls:Callable[[None], nn.Module]=nn.GELU,
    ):
        super().__init__()
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones(ncls, ni))
        # self.gamma_2 = nn.Parameter(1e-4 * torch.ones(ni*ncls))
        self.cls_q = nn.Parameter(torch.zeros((ncls, ni)))
        self.attn = AttentionPool2d(ni, 1, ncls, attn_bias, norm)
        self.norm = norm(ni * ncls)
        self.ffn = nn.Sequential(
            nn.Linear(ni * ncls, int(ni*ffn_expand*ncls)),
            act_cls(),
            nn.Dropout(0.5),
            nn.Linear(int(ni*ffn_expand*ncls), ni*ncls)
        )
        self.inp_per_head = ni/nheads
        self.nheads = nheads
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x:Tensor):
        x = (self.cls_q + self.gamma_1 * self.attn(x, self.cls_q)[0]).flatten(start_dim=1)
        return x + self.ffn(self.norm(x)) # * self.gamma_2

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# Borrowed and modified from https://github.com/pytorch/examples/blob/main/word_language_model/model.py

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
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
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, n_fft, n_rri_inp, dropout=0.1, device="cpu", multiquery=False):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
                              'lower.') from e

        self.pos_encoder = PositionalEncoding(ninp, dropout, device=device)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.rri_conv_net = nn.Sequential(
            # nn.BatchNorm1d(1),
            torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            torch.nn.Conv1d(in_channels=32, out_channels=n_rri_inp, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(n_rri_inp),
        )

        self.rri_pos_encoder = PositionalEncoding(n_rri_inp, dropout, device=device)
        rri_encoder_layers = TransformerEncoderLayer(n_rri_inp, nhead, nhid, dropout)
        self.rri_transformer = TransformerEncoder(rri_encoder_layers, nlayers)
        self.rri_attention_pooling = lambda x: torch.mean(x, dim=-2) # LearnedAggregation(n_rri_inp, ncls=ntoken if multiquery else 1)

        self.ninp = ninp

        self.stft_window = torch.hann_window(n_fft, device=device)
        self.stft_layer_norm = nn.LayerNorm(16)
        self.stft_expand_layer = nn.Sequential(
            nn.Linear(16, ninp),
            nn.ReLU(),
            nn.LayerNorm(ninp))

        self.attention_pooling = lambda x: torch.mean(x, dim=-2) # LearnedAggregation(ninp, ncls=ntoken if multiquery else 1)
        self. layer_norm = nn.LayerNorm(ninp)

        self.decoder1 = nn.Linear((ninp+n_rri_inp)*(ntoken if multiquery else 1), 128)
        self.dropout1 = nn.Dropout(dropout)
        self.decoder2 = nn.Linear(128, ntoken)
        self.n_fft = n_fft

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def fix_transformer_params(self, fix=True):
        for param in self.transformer_encoder.parameters():
            param.requires_grad = (not fix)

        """
        for param in self.rri_transformer.parameters():
            param.requires_grad = (not fix)

        for param in self.rri_conv_net.parameters():
            param.requires_grad = (not fix)
        """

        for param in self.stft_expand_layer.parameters():
            param.requires_grad = (not fix)

        for param in self.stft_layer_norm.parameters():
            param.requires_grad = (not fix)

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder1.bias)
        nn.init.zeros_(self.decoder2.bias)
        nn.init.uniform_(self.decoder1.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder2.weight, -initrange, initrange)

    def stft_and_reshape(self, src):
        ecg_stfts = torch.stft(src, n_fft=self.n_fft, return_complex=True, window=self.stft_window)

        # Cut spectrogram
        src = torch.log(torch.abs(ecg_stfts))[:, 2:18, :]
        # src[:, 68:, :] = 0
        # src[:, :4, :] = 0

        src = torch.permute(src, [2, 0, 1])
        src = self.stft_layer_norm(src[:, :, :self.ninp])
        src = self.stft_expand_layer(src)

        # Layer norm
        # src = (src - torch.mean(src, dim=-1)[:, :, None])/torch.std(src, dim=-1)[:, :, None]

        # Batch norm
        # src = self.spectrogram_bn(torch.unsqueeze(src, dim=1))[:, 0, :, :]
        return src

    def forward(self, src, rri):

        src = self.stft_and_reshape(src)
        src = self.pos_encoder(src)

        rri = torch.unsqueeze(rri, dim=1)
        rri = self.rri_conv_net(rri)

        rri = torch.permute(rri, [2, 0, 1])

        output = self.transformer_encoder(src)
        rri_output = self.rri_transformer(rri)

        rri_output = torch.transpose(rri_output, 0, 1)
        output = torch.transpose(output, 0, 1)

        output = self.attention_pooling(output)
        rri_output = self.rri_attention_pooling(rri_output)

        output = torch.concat([output, rri_output], dim=-1)

        output = self.decoder1(output)
        output = nn.functional.relu(output)
        output = self.dropout1(output)

        output = self.decoder2(output)

        # output = self.sigmoid(output)

        return output
