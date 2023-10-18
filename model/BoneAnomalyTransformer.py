# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .attn import AnomalyAttention, AttentionLayer
# from .embed import DataEmbedding
# import math


# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, attn_mask=None):
#         new_x, attn, mask, sigma = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )       # [256,20,512], 后三个[256,8,20,20]，后三个分别对应series, prior, sigma

#         x = x + self.dropout(new_x)     # [256,20,512]
#         y = x = self.norm1(x)           # [256,20,512]
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   # [256,512,20]
#         y = self.dropout(self.conv2(y).transpose(-1, 1))        # [256,100,512]

#         return self.norm2(x + y), attn, mask, sigma


# class Encoder(nn.Module):
#     def __init__(self, attn_layers, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.norm = norm_layer

#     def forward(self, x, attn_mask=None):   # [256,50,512]
#         # x [B, L, D]
#         series_list = []
#         prior_list = []
#         sigma_list = []
#         for attn_layer in self.attn_layers:
#             x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
#             series_list.append(series)
#             prior_list.append(prior)
#             sigma_list.append(sigma)

#         if self.norm is not None:
#             x = self.norm(x)

#         return x, series_list, prior_list, sigma_list

# # 信息增益层
# class InfomationLayer(nn.Module):
#     def __init__(self, hidden_dim, out_dim, window_size):
#         super(InfomationLayer, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.window_size = window_size
#         # self.linear = nn.Linear(hidden_dim, window_size*out_dim, bias=False)
#         self.QLinear = nn.Linear(out_dim, out_dim, bias=False)
#         self.KLinear = nn.Linear(out_dim, out_dim, bias=False)
#         self.VLinear = nn.Linear(out_dim, out_dim, bias=False)


#     def forward(self, x, y):    # x.shape=[256,20,75]; y.shape=[256,5,75]
#         # print("x.shape", x.shape,y.shape)
#         # x = self.linear(x).view(x.shape[0], self.window_size, -1)  # h_end维度与y不同，先映射成相同维度  x.shape=[256,20,75]
#         x = torch.cat([x, y], dim=1)    # x.shape=[256,25,75]
#         Q = self.QLinear(x)
#         K = self.KLinear(x)
#         V = self.VLinear(x)
#         alpha = torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(self.out_dim)
#         # print("Q",Q.shape,"K",K.shape, "V",V.shape,"alpha",alpha.shape)
#         alpha = torch.softmax(alpha, dim=2)     # dim=0?2?
#         out = torch.bmm(alpha, V)
#         return out

# class BoneAnomalyTransformer(nn.Module):
#     def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
#                  dropout=0.0, activation='gelu', output_attention=True):        # e_layers=3
#         super(BoneAnomalyTransformer, self).__init__()
#         self.output_attention = output_attention
#         self.win_size = win_size

#         # Encoding
#         self.embedding = DataEmbedding(enc_in, d_model, dropout)
#         self.embedding_fea = DataEmbedding(win_size, d_model, dropout)

#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
#                         d_model, n_heads),
#                     d_model,
#                     d_ff,
#                     dropout=dropout,
#                     activation=activation
#                 ) for l in range(e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(d_model)
#         )
#         self.encoder_fea = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         AnomalyAttention(enc_in, False, attention_dropout=dropout, output_attention=output_attention),
#                         d_model, n_heads),
#                     d_model,
#                     d_ff,
#                     dropout=dropout,
#                     activation=activation
#                 ) for l in range(e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(d_model)
#         )

#         self.lin1 = nn.Linear(d_model, win_size)

#         self.projection = nn.Linear(d_model, c_out, bias=True)
#         self.projection2 = nn.Linear(c_out*2, c_out, bias=True)

#         self.info_layer = InfomationLayer(d_model, c_out, win_size)

#     def forward(self, x):   # [256,100,50]
#         # 时间维度
#         enc_out = self.embedding(x)     # [256,100,512]
#         enc_out, series, prior, sigmas = self.encoder(enc_out)  # [256,100,512]，后三个均为长度为3的list，list中元素形状为[256,8,100,100]

#         enc_out = self.projection(enc_out)      # [256,100,50]
#         self.enc_out = enc_out
#         if self.output_attention:
#             return enc_out, series, prior, sigmas
#         else:
#             return enc_out  # [B, L, D]

#     def get_info_y(self, y):
#         # print()
#         info_data = self.info_layer(self.enc_out, y)
#         # print("inputdata",info_data.shape)
#         return info_data[:,self.win_size:,:]

#     def get_info_gain(self, y, info_y):
#         # return self.calcShannonEnt(info_y) - self.calcShannonEnt(y)


#         return self.calcShannonEnt(y) - self.calcShannonEnt(info_y)
    
#     def get_info_gain_maning(self, y, info_y):
#         # return self.calcShannonEnt(info_y) - self.calcShannonEnt(y)
#         # print("self.enc_out",self.enc_out.shape)
#         predict=self.enc_out[:,self.win_size:,:]

#         return self.calcShannonEnt(predict) - self.calcShannonEnt(info_y)

#     def calcShannonEnt(self, data, needMean=True):
#         x = data
#         if data.ndim == 3:
#             x = x.permute(0, 2, 1)
#         # print("x.shape",x.shape, x.mean())
#         scale = 10 # 由于x数值本身大致在0.001~0.01之间，直接用x做softmax可能效果不好，设置一个放大倍数
#         # x = torch.softmax(x, dim=-1)  # [256,75,5]
#         # x=torch.sigmoid(x)
#         # x = torch.softmax(x, dim=-1)  # [256,75,5]
#         # kkk=x * scale
#         # print("kkk",kkk.max(),kkk.min(),kkk.mean())
#         x=x.clamp(-0.5,1)
#         x = torch.softmax(x * scale, dim=-1)  # [256,75,5]

#         x = -(x * torch.log2(x))
#         ent = torch.sum(x, dim=-1)
#         if data.ndim == 3 and needMean:
#             ent = torch.mean(ent, dim=1)
#             # ent = torch.mean(ent)
#         # print("ent:",ent)
#         return ent

import torch
import torch.nn as nn
import torch.nn.functional as F


from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding
import math

from visualizer import get_local


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
        )       # [256,20,512], 后三个[256,8,20,20]，后三个分别对应series, prior, sigma

        x = x + self.dropout(new_x)     # [256,20,512]
        y = x = self.norm1(x)           # [256,20,512]
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   # [256,512,20]
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

    def forward(self, x, attn_mask=None):   # [256,50,512]
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list

# 信息增益层
class InfomationLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim, window_size):
        super(InfomationLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.window_size = window_size
        # self.linear = nn.Linear(hidden_dim, window_size*out_dim, bias=False)
        self.QLinear = nn.Linear(out_dim, out_dim, bias=False)
        self.KLinear = nn.Linear(out_dim, out_dim, bias=False)
        self.VLinear = nn.Linear(out_dim, out_dim, bias=False)


    def forward(self, x, y):    # x.shape=[256,150]; y.shape=[256,5,75]
        # x = self.linear(x).view(x.shape[0], self.window_size, -1)  # h_end维度与y不同，先映射成相同维度  x.shape=[256,20,75]
        x = torch.cat([x, y], dim=1)    # x.shape=[256,25,75]
        Q = self.QLinear(x)
        K = self.KLinear(x)
        V = self.VLinear(x)
        alpha = torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(self.out_dim)
        alpha = torch.softmax(alpha, dim=2)     # dim=0?2?
        out = torch.bmm(alpha, V)
        return out

class BoneAnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=1, e_layers=1, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):        # e_layers=3
        super(BoneAnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.win_size = win_size

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.embedding_fea = DataEmbedding(win_size, d_model, dropout)

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
        self.encoder_fea = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(enc_in, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.lin1 = nn.Linear(d_model, win_size)

        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.projection2 = nn.Linear(c_out*2, c_out, bias=True)

        self.info_layer = InfomationLayer(d_model, c_out, win_size)

    # def forward(self, x):   # [256,100,50]
    #     # 时间维度
    #     enc_out = self.embedding(x)     # [256,100,512]
    #     enc_out, series, prior, sigmas = self.encoder(enc_out)  # [256,100,512]，后三个均为长度为3的list，list中元素形状为[256,8,100,100]
    #
    #     enc_out = self.projection(enc_out)      # [256,100,50]
    #     self.enc_out = enc_out
    #     if self.output_attention:
    #         return enc_out, series, prior, sigmas
    #     else:
    #         return enc_out  # [B, L, D]

    # 先空再时
    # def forward(self, x):   # [256,100,50]
    #
    #     # 空间维度
    #     fea_enc_out = self.embedding_fea(x.permute(0, 2, 1))    # [256,50,512]
    #     fea_enc_out, _, _, _ = self.encoder_fea(fea_enc_out)    # [256,50,512]
    #     fea_enc_out = self.lin1(fea_enc_out).permute(0, 2, 1)   # [256,100,50]
    #     # 时间维度
    #     time_enc_out = self.embedding(fea_enc_out)  # [256,100,512]
    #     time_enc_out, series, prior, sigmas = self.encoder(time_enc_out)  # [256,100,512]，后三个均为长度为3的list，list中元素形状为[256,8,100,100]
    #
    #     enc_out = self.projection(time_enc_out)      # [256,100,50]
    #     self.enc_out = enc_out
    #
    #     if self.output_attention:
    #         return enc_out, series, prior, sigmas
    #     else:
    #         return enc_out  # [B, L, D]

    # 时空拼接 目前用这个
    # @get_local('enc_out')
    def forward(self, x):   # [256,20,45]

        # 空间维度
        fea_enc_out = self.embedding_fea(x.permute(0, 2, 1))    #
        fea_enc_out, a, b, c = self.encoder_fea(fea_enc_out)    #
        fea_enc_out = self.lin1(fea_enc_out).permute(0, 2, 1)   # [256,20,50]
        # 时间维度
        time_enc_out = self.embedding(x)  # [256,20,512]
        time_enc_out, series, prior, sigmas = self.encoder(time_enc_out)  # [128,20,512]，后三个均为长度为3的list，list中元素形状为[128,8,20,20]
        time_enc_out = self.projection(time_enc_out)      # [128,20,45]

        enc_out = torch.cat([fea_enc_out, time_enc_out], dim=2)  # [128,20,90]
        enc_out = self.projection2(enc_out)
        self.enc_out = enc_out

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]

    def get_info_y(self, y):
        info_data = self.info_layer(self.enc_out, y)
        return info_data[:,self.win_size:,:]

    def get_info_gain(self, y, info_y):
        # return self.calcShannonEnt(info_y) - self.calcShannonEnt(y)
        predict=self.enc_out[:,self.win_size:,:]
        return self.calcShannonEnt(predict) - self.calcShannonEnt(info_y)

    def calcShannonEnt(self, data, needMean=True):
        x = data
        if data.ndim == 3:
            x = x.permute(0, 2, 1)
        scale = 10  # 由于x数值本身大致在0.001~0.01之间，直接用x做softmax可能效果不好，设置一个放大倍数
        # x = torch.softmax(x, dim=-1)  # [256,75,5]
        x = torch.softmax(x * scale, dim=-1)  # [256,75,5]
        x = -(x * torch.log2(x))
        ent = torch.sum(x, dim=-1)
        if data.ndim == 3 and needMean:
            ent = torch.mean(ent, dim=1)
            # ent = torch.mean(ent)
        return ent