import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from os import listdir, makedirs, path
from pickle import dump


class BoneTimeDataset(Dataset):
    def __init__(self, raw_data, mode='train', config = None):
        self.raw_data = raw_data

        self.config = config
        self.mode = mode
        x_all = []
        y_all = []
        labels_all = []
        for i in range(len(raw_data)):
            data_seg = raw_data[i]      # [V+1, T], eg:[50+1, 78], T可能不同，代表单个动作的帧数
            x_data = data_seg[:-1]
            labels = data_seg[-1]

            data = x_data
            # to tensor
            data = torch.tensor(data).double()
            labels = torch.tensor(labels).double()

            x, y, labels = self.process(data, labels)
            if x != None:
                x_all.append(x)
                y_all.append(y)
                labels_all.append(labels)
        self.x = torch.cat(x_all)
        self.y = torch.cat(y_all)
        self.labels = torch.cat(labels_all)

        if self.mode != 'train':
            # save {dataset}_label.pkl
            output_folder = "datasets/ntu/processed"
            with open(path.join(output_folder, "NTU_test_label.pkl"), "wb") as file:
                dump(labels, file)
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []
        slide_win, slide_stride, predict_win = [self.config[k] for k
            in ['slide_win', 'slide_stride', 'predict_win']
        ]

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride)
        
        for i in rang:
            if i+predict_win > total_time_len:     # 预测y的窗口已超出数据长度
                break
            ft = data[:, i-slide_win:i]
            tar = data[:, i:i+predict_win]    # 骨骼点数据预测的y可能是下面的若干帧，而不是1帧

            x_arr.append(ft)
            y_arr.append(tar)   # 用前一段长度为5的样本，预测下一个样本

            labels_arr.append(labels[i])        # 因为这里是对每个seg处理的，而每个300帧的seg标签都一样，所以不用考虑预测y窗口内可能存在不同label的情况

        x, y, labels = None, None, None
        if len(x_arr) != 0:
            x = torch.stack(x_arr).contiguous()     # x_arr转tensor再深拷贝至x; x.shape:[1560,27,5]（train的数据，下同）
            y = torch.stack(y_arr).contiguous()     # y.shape:[1560,27]

            labels = torch.Tensor(labels_arr).contiguous()  #labels.shape:[1560]
        
        return x, y, labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        label = self.labels[idx].double()

        return feature, y, label





