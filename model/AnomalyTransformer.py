import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )       # [256,100,512], 后三个[256,8,100,100]，后三个分别对应series, prior, sigma

        x = x + self.dropout(new_x)     # [256,100,512]
        y = x = self.norm1(x)           # [256,100,512]
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   # [256,512,100]
        y = self.dropout(self.conv2(y).transpose(-1, 1))        # [256,100,512]
        # x1 = x + self.dropout(new_x)  # [256,100,512]
        # x2 = self.norm1(x1)  # [256,100,512]
        # y = x2.clone()  # [256,100,512]
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # [256,512,100]
        # y = self.dropout(self.conv2(y).transpose(-1, 1))  # [256,100,512]

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):   # [256,100,512]
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            # x = torch.Tensor(256,100,512).to(torch.device("cuda:4"))
            # series = torch.Tensor(256,8,100,100).to(torch.device("cuda:4"))
            # prior = torch.Tensor(256,8,100,100).to(torch.device("cuda:4"))
            # sigma = torch.Tensor(256,8,100,100).to(torch.device("cuda:4"))
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        # x_list = [x]
        # for i, attn_layer in enumerate(self.attn_layers):     # 3层，上一层的输出x作为下一层的输入x
        #     nx, series, prior, sigma = attn_layer(x_list[i], attn_mask=attn_mask)    # [256,100,512], 后三个都是[256,8,100,100]
        #     series_list.append(series)
        #     prior_list.append(prior)
        #     sigma_list.append(sigma)
        #     x_list.append(nx)
        # xx = x_list[-1]
        #
        # xx = torch.Tensor(256,100,512).to(torch.device("cuda:4"))
        # series = torch.Tensor(256,8,100,100).to(torch.device("cuda:4"))
        # series_list.append(series)
        # series_list.append(series)
        # series_list.append(series)
        # prior_list.append(series)
        # prior_list.append(series)
        # prior_list.append(series)
        # sigma_list.append(series)
        # sigma_list.append(series)
        # sigma_list.append(series)
        #
        # if self.norm is not None:
        #     xx = self.norm(xx)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):        # e_layers=3
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):   # [256,100,50]
        enc_out = self.embedding(x)     # [256,100,512]
        enc_out, series, prior, sigmas = self.encoder(enc_out)  # [256,100,512]，后三个均为长度为3的list，list中元素形状为[256,8,100,100]

        enc_out = self.projection(enc_out)      # [256,100,50]

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
