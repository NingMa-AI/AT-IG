import os
import sys
from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
from pickle import dump
import numpy as np
import pandas as pd

from args import get_parser

import pickle
import torch
from BoneTimeDataset import BoneTimeDataset
from sklearn.preprocessing import MinMaxScaler
import json


def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(
        path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset, dataset_folder=None, output_folder=None, abnormal_ratio=0.1):
    """ Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly) """

    if dataset == "SMD":
        dataset_folder = "datasets/ServerMachineDataset"
        output_folder = "datasets/ServerMachineDataset/processed"
        makedirs(output_folder, exist_ok=True)
        file_list = listdir(path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save(
                    "train",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
    elif dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "datasets/data"
        output_folder = "datasets/data/processed"
        makedirs(output_folder, exist_ok=True)
        with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True
            labels.extend(label)

        labels = np.asarray(labels)
        print(dataset, "test_label", labels.shape)

        with open(path.join(output_folder, dataset + "_" + "test_label" + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(path.join(dataset_folder, category, filename + ".npy"))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ["train", "test"]:
            concatenate_and_save(c)
    elif dataset == 'NTU':
        dataset_folder = "../../data/ntu/xview/"
        output_folder = "datasets/ntu/processed21"

        makedirs(output_folder, exist_ok=True)
        # anomaly_detection_label = [1, 26]   # [吃东西,跳起来]
        # anomaly_detection_label = [26, 42]   # [跳起来,跌倒]
        # anomaly_detection_label = [5, 6]   # [拾起,扔]
        # anomaly_detection_label = [5, 7]   # [拾起,坐下]
        # anomaly_detection_label = [26, 5]   # [跳起来,拾起]
        # anomaly_detection_label = [26, 6]   # [跳起来,扔]
        # anomaly_detection_label = [26, 7]   # [跳起来,坐下]
        anomaly_detection_label = [26, 49]
        frame_sample_interval = 1   # 间隔抽帧步长，1表示连续，即不抽帧
        predict_interval = 0        # 预测y间隔x几帧，0表示连续，即不间隔
        predict_win = 5             # y窗口长度，默认为1
        abnormal_ratio = 0.14

        save_data(dataset_folder, output_folder, anomaly_detection_label,frame_sample_interval,predict_interval,predict_win,abnormal_ratio, False)
        # save_data(dataset_folder, output_folder, anomaly_detection_label,frame_sample_interval,predict_interval,predict_win, True)    # OOD
    elif dataset == 'BONE':
        # dataset_folder = "../../data/anomaly_detection/train"
        # output_folder = "datasets/bone/processed3"
        # abnormal_ratio = 0.1

        dataset_folder = "../../data/anomaly_detection/val"  # 真实婴儿测试数据
        output_folder = "datasets/bone/processed3_test"     # 真实婴儿测试数据

        makedirs(output_folder, exist_ok=True)

        predict_win = 5

        save_data_bone(dataset_folder, output_folder, predict_win, abnormal_ratio)

def save_data(root_path, out_path, anomaly_detection_label,frame_sample_interval,predict_interval,predict_win, abnormal_ratio, is_train=True, use_mmap=False):
    # data: N C V T M
    label_path = root_path + ('train_label.pkl' if is_train else 'val_label.pkl')
    data_path = root_path + ('train_data_joint.npy' if is_train else 'val_data_joint.npy')
    data_frame_path = root_path + ('train_num_frame.npy' if is_train else 'val_num_frame.npy')
    try:
        with open(label_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')

    # load data
    if use_mmap:
        data_orig = np.load(data_path, mmap_mode='r')
        data_frame = np.load(data_frame_path, mmap_mode='r')
    else:
        data_orig = np.load(data_path)  # data.shape=[18932,3,300,25,2]; label.shape=[18932]
        data_frame = np.load(data_frame_path)

    # data_orig中每行都是一个300帧的动作，但其中动作不足300帧的会重复拼接，根据data_frame，提取单个完整的连续动作
    data = []
    # min_frame = int(data_frame.min())   # 所有动作都
    N, C, T, V, M  = data_orig.shape

    normal_label, abnormal_label= anomaly_detection_label[0], anomaly_detection_label[1]
    abnormal_num = int(np.where(np.array(label) == abnormal_label)[0].size * abnormal_ratio)   # 正常样本与异常样本比例，具体比例还要看划分滑动窗口后，会打印输出
    count,count_train = 0, 0
    # for i in range(data_frame.size):
    #     # 训练集中只有正常样本，测试集中有正常和异常样本
    #     if label[i] == normal_label or (not is_train and label[i] == abnormal_label):
    #         # data_orig：N C T V M（N, 3, 300, 25, 2）
    #         x = data_orig[i,:2,:int(data_frame[i]),:,0].reshape(1,2,-1,V).transpose(0,3,1,2) # N V C T (N=1)
    #         x = x.reshape(-1,x.shape[-1])     # [V*C,T], 将xy所在维度合并到V维度，即将x、y各作为一个特征，特征数变为25*2=50
    #         # 间隔抽帧
    #         sample_indices = np.arange(0, x.shape[-1], frame_sample_interval)
    #         x = x[:, sample_indices]
    #         t_len = x.shape[-1]
    #
    #         if label[i] == normal_label and t_len>=args.lookback:
    #             print(t_len)
    #             labels = np.zeros((1,t_len))
    #             x = np.concatenate((x, labels), axis=0)  # [V*C + 1,T]
    #             data.append(x)
    #         elif label[i] == abnormal_label and t_len>=args.lookback and count < abnormal_num:
    #             print('abnormal:', t_len)
    #             labels = np.ones((1,t_len))
    #             x = np.concatenate((x, labels), axis=0)  # [V*C + 1,T]
    #             data.append(x)
    #             count = count + 1
    for i in range(data_frame.size):
        # 训练集和测试集都有正常和异常数据，但训练集中异常数据较少
        if label[i] == normal_label or label[i] == abnormal_label:
            # data_orig：N C T V M（N, 3, 300, 25, 2）
            # x = data_orig[i,:2,:int(data_frame[i]),:,0].reshape(1,2,-1,V).transpose(0,3,1,2) # N V C T (N=1)
            x = data_orig[i,:,:int(data_frame[i]),:,0].reshape(1,3,-1,V).transpose(0,3,1,2) # N V C T (N=1)     # 保留z坐标
            x = x.reshape(-1,x.shape[-1])     # [V*C,T], 将xy所在维度合并到V维度，即将x、y各作为一个特征，特征数变为25*2=50 # 加上z坐标变成25*3=75
            # 间隔抽帧
            sample_indices = np.arange(0, x.shape[-1], frame_sample_interval)
            x = x[:, sample_indices]
            t_len = x.shape[-1]
            is_normal = label[i] == normal_label

            if is_normal and t_len>=args.lookback:  # 正常样本
                # print(t_len)
                labels = np.zeros((1,t_len))
                x = np.concatenate((x, labels), axis=0)  # [V*C + 1,T]
                data.append(x)
            elif not is_normal and t_len>=args.lookback:    # 异常样本
                add_flag = False
                if is_train and count_train < abnormal_num:   # train、test中异常数据占比可能不同，用abnormal_num控制
                    add_flag = True
                    count_train += 1
                elif not is_train and count < abnormal_num:
                    add_flag = True
                    count +=1
                if add_flag:
                    # print('abnormal:', t_len)
                    labels = np.ones((1,t_len))
                    x = np.concatenate((x, labels), axis=0)  # [V*C + 1,T]
                    data.append(x)
    # data.shape = [N, V+1, T]，其中V=50(即原来的V*C), +1为label, T各不相同（单个动作帧数）

    # 划分等长的预测窗口，然后拼接起来，[x1,y1,x2,y2....]，其中x.shape=[100,V],y.shape=[1,V]，因此每101个长度为一组数据，后面SlidingWindowDataset.__getitem__中需要对应处理，怎么取出一组数据
    cfg = {
        'slide_win': args.lookback,
        'slide_stride': 1,
        'predict_interval': predict_interval,
        'predict_win': predict_win
    }
    dataset, labels = split_window(data, config=cfg)

    # save
    category = 'train'
    # save {dataset}_label.pkl
    if not is_train:
        with open(path.join(out_path, "NTU_test_label.pkl"), "wb") as file:
            dump(labels, file)
        category = 'test'
    # save {dataset}_train.pkl or {dataset}_test.pkl
    with open(path.join(out_path, "NTU_" + category + ".pkl"), "wb") as file:
        dump(dataset, file)
    # AD test和train同一个数据集，将test同时存为train
    with open(path.join(out_path, "NTU_train.pkl"), "wb") as file:
        dump(dataset, file)

    [normal_len,abnormal_len] = [np.where(labels==i)[0].shape[0] for i in [0,1]]
    print(category,"_normal_len:",normal_len)
    print(category,"_abnormal_len:",abnormal_len)
    print(category,"_abnormal_rate:",abnormal_len/(normal_len+abnormal_len))

def save_data_bone(root_path, out_path, predict_win, abnormal_ratio=None):
    # ../../data/anomaly_detection/test
    data = []
    file_names = []  # 记录生成数据集使用的文件名
    if abnormal_ratio!=0:
        abnormal_cnt = 0    # 先计算一下异常数据的数量
        for root_dir, sub_dir, files in os.walk(root_path):
            files.sort()
            for file_name in files:
                if file_name.find(args.abnormal_label) != -1:
                    abnormal_cnt = abnormal_cnt+1
        normal_num = abnormal_cnt / abnormal_ratio * (1-abnormal_ratio)

        # print("***************************",abnormal_cnt,normal_num)
        print(f"abnormal_num: {abnormal_cnt}, normal_num: {normal_num}, abnormal_ratio: {abnormal_cnt/(abnormal_cnt+normal_num)}")
    else:   # None表示全用
        normal_num = sys.maxsize
    for root_dir, sub_dir, files in os.walk(root_path):
        files.sort()
        cnt = 0
        for file_name in files:
            if file_name.endswith(".json"):
                if file_name.find(args.abnormal_label) == -1:
                    if cnt > normal_num: continue 
                    cnt = cnt+1

                # 处理该文件数据
                file_names.append(file_name)
                # data_orig: T V C M
                data_orig = pd.read_json(root_path+'/'+file_name)
                data_orig = torch.tensor(data_orig['data'].values.tolist()).double()        # [300,25,3,1]
                data_orig = data_orig[:, [6,7,11,14,19,22,3, 4], 3:6].squeeze().permute(1, 2, 0).contiguous() # 0:3 x,y,z   3:6: angle
                #11 feature [6,7,11,14,19,22,3, 4, 21,24,1] 
                # 6 featrue [6,7,11,14,19,22]   
                # 8 [6,7,11,14,19,22,3, 4]       
                # print("********************",data_orig.shape)    #去除置信度，改为V C T [25,2,300]
                # exit()
                data_orig = data_orig.view(-1, data_orig.shape[-1])  #[V*C, T]  [50,300]
                # 添加labels
                if file_name.find(args.abnormal_label) != -1:  #异常样本label==1
                    labels = torch.ones(1, data_orig.shape[-1])
                else:
                    labels = torch.zeros(1, data_orig.shape[-1])
                data_orig = torch.cat((data_orig.float(), labels))      # [V*C+1, T] [51,300]
                data.append(data_orig)
    # data.shape = [N, V+1, T]，其中V=50(即原来的V*C), +1为label, T各不相同（单个动作帧数）

    # 存储上述生成数据使用的文件名（后续预测时需要对应婴儿）
    file = open(out_path + '_file_names', 'w')
    file.write(json.dumps(file_names))
    file.close()

    # 划分等长的预测窗口，然后拼接起来，[x1,y1,x2,y2....]，其中x.shape=[100,V],y.shape=[1,V]，因此每101个长度为一组数据，后面SlidingWindowDataset.__getitem__中需要对应处理，怎么取出一组数据
    cfg = {
        'slide_win': args.lookback,
        'slide_stride': 5,
        'predict_win': predict_win
    }
    dataset, labels = split_window(data, config=cfg)

    # save
    # save {dataset}_test_label.pkl
    with open(path.join(out_path, "BONE_test_label.pkl"), "wb") as file:
        dump(labels, file)
    # save {dataset}_train.pkl or {dataset}_test.pkl
    with open(path.join(out_path, "BONE_test.pkl"), "wb") as file:
        dump(dataset, file)
    # AD test和train同一个数据集，将test同时存为train
    with open(path.join(out_path, "BONE_train.pkl"), "wb") as file:
        dump(dataset, file)

    [normal_len, abnormal_len] = [np.where(labels == i)[0].shape[0] for i in [0, 1]]
    print("normal_len:", normal_len)
    print("abnormal_len:", abnormal_len)
    print("abnormal_rate:", abnormal_len / (normal_len + abnormal_len))

def split_window(raw_data, config=None):

    x_all = []
    labels_all = []
    for i in range(len(raw_data)):
        data_seg = raw_data[i]  # [V+1, T], eg:[50+1, 78], T可能不同，代表单个动作的帧数
        x_data = data_seg[:-1]
        labels = data_seg[-1]

        data = x_data

        # to tensor
        if not torch.is_tensor(data):
            data = torch.tensor(data).double()
            labels = torch.tensor(labels).double()

        x, labels = process(data, labels, config)
        if labels != None:
            x_all.append(x)
            labels_all.append(labels)
    x = torch.cat(x_all)
    labels = torch.cat(labels_all)

    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels)
    return x, labels

def process(data, labels, config):
    labels_arr = []
    x_data = None
    slide_win, slide_stride, predict_win = [config[k] for k
                                            in ['slide_win', 'slide_stride', 'predict_win']
                                            ]

    node_num, total_time_len = data.shape

    rang = range(slide_win, total_time_len, slide_stride)

    for i in rang:
        if i + predict_win > total_time_len:  # 预测y的窗口已超出数据长度
            break
        ft = data[:, i - slide_win:i].permute(1,0)
        tar = data[:, i:i + predict_win].permute(1,0)  # y可能会间隔几帧，predict_interval=0时不间隔, predict_win表示y的窗口长度
        if x_data == None:
            x_data = torch.cat([ft,tar])
        else:
            x_data = torch.cat([x_data,ft,tar])

        labels_arr.append(labels[i])  # 因为这里是对每个seg处理的，而每个300帧的seg标签都一样，所以不用考虑预测y窗口内可能存在不同label的情况

    labels = None
    if len(labels_arr) != 0:
        labels = torch.Tensor(labels_arr).contiguous()  # labels.shape:[1560]

    return x_data, labels

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    # load_data(ds)

    abnormal_ratio = args.abnormal_ratio

    # dataset_folder = "../../data/anomaly_detection/train"
    dataset_folder = args.dataset_folder+'/train'
    output_folder = args.output_folder
    makedirs(output_folder, exist_ok=True)
    save_data_bone(dataset_folder, output_folder, args.predict_win, abnormal_ratio)


    # dataset_folder = "../../data/anomaly_detection/val"  # 真实婴儿测试数据
    # output_folder = "datasets/bone/processed3_test"  # 真实婴儿测试数据
    dataset_folder = args.dataset_folder+'/test'
    output_folder = args.output_folder+'_test'
    makedirs(output_folder, exist_ok=True)
    save_data_bone(dataset_folder, output_folder, args.predict_win, abnormal_ratio)
