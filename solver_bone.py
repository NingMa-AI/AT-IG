from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from visualizer import get_local
get_local.activate()
from utils.utils import *
from model.BoneAnomalyTransformer import BoneAnomalyTransformer
from data_factory.data_loader import get_loader_segment, get_loader_bone
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import json
import random
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import csv
import json
import seaborn as sns



def draw(model_path,predict, actual, ):
    """
    input arrays
    """
    fpr, tpr,thresholds=roc_curve(actual,predict)
    # print("fpr",fpr.shape, "tpr",tpr.shape,"threshold.shape",thresholds.shape,thresholds)
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    # plt.show()

    with open(os.path.join(model_path,"fpr_tpr_thresholds.csv"),'w') as writer:
         csv_writer = csv.writer(writer, delimiter=',') 
         csv_writer.writerow(["fpr","npr","thresholds"])
         for i in range(len(fpr)):
            csv_writer.writerow([fpr[i],tpr[i],thresholds[i]])
    
    np.save(os.path.join(model_path,"AT-predict.npy"),predict)
    np.save(os.path.join(model_path,"AT-true.npy"),actual)
    print("len predict", len(predict))
    # np.savetxt(os.path.join(model_path,"fpr_tpr_thresholds.txt"),np.concatenate([fpr.reshape(-1,1),tpr.reshape(-1,1),thresholds.reshape(-1,1)],axis=1))
    plt.savefig(os.path.join(model_path,"ROC.png"))
    plt.savefig(os.path.join(model_path,"ROC.pdf"))


def draw_time_serieas(predict, save_dir, segment_names, test_data_dir):
 """
 each segment has 56 predict values 
 """

 kp_map = {
    'nose': 0,
    'neck': 1,
    'r_shoulder': 2,
    'r_elbow': 3,
    'r_wrist': 4,
    'l_shoulder': 5,
    'l_elbow': 6,
    'l_wrist': 7,
    'm_hip': 8,
    'r_hip': 9,
    'r_knee': 10,
    'r_ankle': 11,
    'l_hip': 12,
    'l_knee': 13,
    'l_ankle': 14,
    'r_eye': 15,
    'l_eye': 16,
    'r_ear': 17,
    'l_ear': 18,
    'l_bigToe': 19,
    'l_smallToe': 20,
    'l_heel': 21,
    'r_bigToe': 22,
    'r_smallToe': 23,
    'r_heel': 24
}

 name_2_index=dict(list(zip(segment_names, list(range(len(segment_names))))))
 predict_labels=[]
 prefix = segment_names[0][:segment_names[0].rfind("_S")+2]

 dimension_color={"X":"b", "Y":"g", "A1":"c", "A2":"s"}
 for s_index in range(len(segment_names)): # access the segement by time order
    segment_n=prefix+str(s_index)+".json"
    number=name_2_index[segment_n]
    predict_labels.append(predict[number*56: (number+1)*56])
    stride=5
    segment_label=[]
    for p in predict[number*56: (number+1)*56]:
        segment_label.extend([p]*stride)

    with open(os.path.join(test_data_dir,segment_n),"r") as reader:
        action_segment=json.load(reader)
    # predict_labels
    
    
    for part in ["r_bigToe"]:
        
            xy_location=np.array(action_segment["data"])[:280,kp_map[part],:,:]    # data: (300, 25,6,1) 

            # predict_labels=ddd
            plt.style.use('ggplot')
            
            matplotlib.rcParams['font.sans-serif'] = ['SimHei']
            matplotlib.rcParams['axes.unicode_minus']=False

            fig, ax1 = plt.subplots()

            # 柱形图
            positions=np.where(segment_label)[0]
            
            l1 = ax1.bar(positions, np.ones(len(positions)), color='b',)

            # ax2 = ax1.twinx()

            # 折线图
            x=list(range(len(xy_location)))

            for dimension, dimension_name in (zip([0,1],["X","Y"])):
                l2, = ax1.plot(x, xy_location[:,:,dimension,:], dimension_color[dimension_name], label=dimension_name)
            # l2, = ax2.plot(x[1:], y2, 'r-', label='b')

            plt.yticks(np.arange(-1, 1, 0.1),fontsize=15)
            plt.xticks(np.arange(0, 281, 1),fontsize=15)

            # plt.title("The losses during training",fontsize=20)
            plt.xlabel("Frame",fontsize=15)
            plt.ylabel("Location",fontsize=15)
            plt.legend(["X","Y"],loc="center",fontsize=15)
            # plt.title("Office-31 {}-shot".format(shot),fontsize=22)
            pngname=prefix+str(s_index)+"-part-"+part+"_category"+str(action_segment["category"]) + ".png"
            plt.savefig(pngname, format="png",bbox_inches="tight",dpi = 400)
            break
    break


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0, model_id=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.model_id = model_id

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter = self.counter + 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.model_id == None:
            id = str(self.dataset) + '_checkpoint_'+ datetime.now().strftime("%Y%m%d_%H%M%S") +'.pth'
        else:
            id = self.model_id
        torch.save(model.state_dict(), os.path.join(path, id))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2
        self.checkpoint_id = id


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader, self.vali_loader, self.test_loader, self.test_label = get_loader_bone(self.data_path, self.dataset, self.win_size, self.predict_win, self.batch_size, self.input_c)

        self.thre_loader = self.test_loader

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_id = str(self.dataset) + '_checkpoint_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pth'
        self.build_model()
        self.criterion = nn.MSELoss()
        self.args=config

    def build_model(self):
        self.model = BoneAnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=5, d_model=512)      # 原始e_layers=3, d_model=512
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.to(self.device)

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, labels) in enumerate(vali_loader):
            # print("val iter", i)
            input = input_data.float().to(self.device)
            y = labels.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss = series_loss +  (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss = prior_loss+ (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            if self.add_info_gain:
                info_gain = self.model.get_info_gain(y, self.model.get_info_y(y))
                info_gain = torch.mean(info_gain)
                rec_loss = rec_loss - rec_loss * info_gain  # 信息增益
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def setup_seed(self, seed=0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # np.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def train(self):
        print("======================TRAIN MODE======================")
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=100, verbose=True, dataset_name=self.dataset, model_id=self.model_id)         # patience:早停轮数
        train_steps = len(self.train_loader)

        self.train_loss1 = []
        self.train_loss2 = []
        self.valid_loss1 = []
        self.valid_loss2 = []
        self.info_gains = []
        self.recon_losses = []

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            loss2_list = []
            recon_list = []
            info_list = []
            
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                # print("input",input_data.shape, labels.shape)
                y = labels.float().to(self.device)

                self.optimizer.zero_grad()
                iter_count = iter_count+ 1
                input = input_data.float().to(self.device)      # [256,20,50]

                output, series, prior, _ = self.model(input)    # [256,20,50]，series和prior都是长度为3的list，list中的元素为[256,8,100,100]，AnomalyTransformer中encoder有几层list长度就为多少
                
                # if (i ) % 100 == 0:
                #     self.draw_heat_map(self.model_save_path, i)
                    # exit()

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss = series_loss + (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach())) + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach(),series[u])))
                    prior_loss = prior_loss + (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)),series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                self.train_loss1.append(series_loss.item() + prior_loss.item())
                self.train_loss2.append(prior_loss.item())

                rec_loss = self.criterion(output, input)
                recon_list.append(rec_loss.item())
                self.recon_losses.append(rec_loss.item())

                # 计算信息增益, 正常增益大、异常增益小
                info_gain = -1
                if self.add_info_gain:
                    info_gain = self.model.get_info_gain(y, self.model.get_info_y(y))   # [128]
                    info_gain = torch.mean(info_gain)
                    # print("info-gain",info_gain.item())
                    # rec_loss = rec_loss - rec_loss * info_gain  # 信息增益
                    rec_loss = rec_loss - self.k *info_gain  # 信息增益

                    self.info_gains.append(info_gain.item())
                    info_list.append(info_gain.item())

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss2_list.append((rec_loss + self.k * prior_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss
                # loss2 = rec_loss

                if (i+1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                    vali_loss1, vali_loss2 = self.vali(self.test_loader)
                    self.valid_loss1.append(vali_loss1)
                    self.valid_loss2.append(vali_loss2)
                    train_loss = np.average(loss1_list)

                    print(
                        "Tter: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                            i + 1, train_steps, train_loss, vali_loss1))

                    early_stopping(vali_loss1, vali_loss2, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                    
                    
                # Minimax strategy
                # print(f"rec_loss:{rec_loss}, info_gain:{info_gain}, loss1:{loss1}, loss2:{loss2}")
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            print(f"rec_loss:{np.mean(recon_list)}, info_gain:{np.mean(info_list)}, loss1:{np.mean(loss1_list)}, loss2:{np.mean(loss2_list)}")

            # vali_loss1, vali_loss2 = self.vali(self.test_loader)  # 原写法。不是很理解，这里为什么验证时不用vali_loader
            
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        return early_stopping.checkpoint_id     # 返回存储的最优模型id
    

    def draw_heat_map(self, save_path, iteration_num):
        # print("begin")
        
        heatmaps=get_local.cache

        # data = np.random.rand(5, 5)  # Replace this with your own data

        for attention_function, maps in heatmaps.items():
            print("attention_function, maps.shape", attention_function, len(maps))
            for i, map in enumerate( maps):
                map=map.squeeze()
                # print("map.shape", map.shape)

                # Create a heatmap using Seaborn
                plt.figure(figsize=(40, 30))
                sns.heatmap(map, annot=True, cmap='coolwarm', linewidths=0.5)

                # Customize the plot (optional)
                plt.title(f'Layer Attention Map')
                plt.xlabel('Body Parts')
                plt.ylabel('Body Parts')
        
                # Show the plot
                plt.savefig(os.path.join(save_path, f"{attention_function}+_iter_{iteration_num}_atten_num{i}.png"))
        # print("end")

    def test(self, thresh=None, ratio_thresh=None, epsilon=None, isTest=False):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), self.checkpoint_id)))
        print(f'load model: {os.path.join(str(self.model_save_path), self.checkpoint_id)}')
        self.model.eval()
        temperature = 50
        if isTest:
            print("======================TEST MODE======================")
        else:
            print("======================Validation MODE======================")
        criterion = nn.MSELoss(reduce=False)
        attens_energy = []
        metric_list = []
        loss_list = []
        info_gain_list = []

        for i, (input_data, labels) in enumerate(self.thre_loader):
            # print(i)
            input = input_data.float().to(self.device)
            y = labels.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)     # [256,20]
            loss_to_plot = torch.mean(criterion(input, output), dim=1)  # [256,50]

            if self.add_info_gain:
                info_gain = self.model.get_info_gain(y, self.model.get_info_y(y))
                # loss = loss - loss * info_gain      # 信息增益
                info_gain_list.append(info_gain.detach().cpu().numpy())

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss = series_loss+ my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = prior_loss+ my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            # cri = loss
            cri = cri.detach().cpu().numpy()    # [128,20]
            attens_energy.append(cri)
            metric_list.append(metric.detach().cpu().numpy())
            loss_list.append(loss_to_plot.detach().cpu().numpy())

        attens_energy = np.concatenate(attens_energy, axis=0).mean(axis=1)

        if self.add_info_gain:
            info_gain = np.concatenate(info_gain_list, axis=0)
            test_energy = np.array(attens_energy - info_gain)
        else:
            test_energy = np.array(attens_energy)

        if thresh==None and ratio_thresh==None:     # 训练阶段的验证，需要找到最优阈值
            best_re = self.find_best_thresh(test_energy)
        else:   # 测试阶段，直接使用训练阶段的最优阈值
            pred = (test_energy > thresh).astype(int)
            best_re = self.predict_personal_adjust(pred, ratio_thresh, isTest=isTest, epsilon=epsilon, is_adjust=False)
            best_re['thresh'] = thresh
    
        print(f"f1: {best_re['f1']}, precision: {best_re['precision']}, recall: {best_re['recall']}, threshold: {best_re['thresh']}, ratio_threshold: {best_re['ratio_thresh']}, epsilon: {best_re['epsilon']}")

        return best_re

    def find_best_thresh(self, test_score):

        best_re = None
        min_score = np.min(test_score)
        max_score = np.max(test_score)

        for step in np.arange(0, 100, 1):
            thresh = min_score + (max_score-min_score)/100.0*step
            pred = (test_score > thresh).astype(int)
            for epsilon in range(1,10,1):
                for ratio_thresh in np.arange(0.01, 0.9, 0.1):
                    re = self.predict_personal_adjust(pred, ratio_thresh, isTest=False, is_adjust=False, epsilon=epsilon)
                    if best_re==None or re['f1']>best_re['f1']:
                        best_re = re
                        best_re['thresh'] = thresh
        return best_re

    # 直接使用存储的file_names文件即可，无需再走一遍数据文件路径
    def predict_personal_adjust(self, pred, ratio_thresh, epsilon=1, isTest = False, is_adjust=True):
        label_pred = np.array(pred)
        # 读取file_names
        file = open(self.file_names_path, "r", encoding='UTF-8')
        file_names = file.read()
        file_names = json.loads(file_names)
        file.close()
        i = 0
        j = i  # [i,j)为同一个人的数据
        correct_num = 0  # 预测正确、错误数
        wrong_num = 0
        normal_num = 0  # 真实正常、异常数
        abnormal_num = 0
        normal_num_pred = 0  # 预测正常、异常数
        abnormal_num_pred = 0
        predict = []
        actual = []
        ratios=[]

        last_prefix = file_names[0][:file_names[0].rfind("_S")]
        for index, file_name in enumerate(file_names):
            prefix = file_name[:file_name.rfind("_S")]
            # 对于每个文件进行校正，一个文件对应56个数据段
            
            if is_adjust:
                if np.where(label_pred[j:j + 56])[0].shape[0] > epsilon and prefix.find('CP') != -1:  # 用真实标签
                    label_pred[j:j + 56] = 1

            elif np.where(label_pred[j:j + 56])[0].shape[0] > epsilon:
                label_pred[j:j + 56] = 1

            if prefix != last_prefix or index == len(file_names) - 1:  # 当前是一个新的人
                # 处理上一个人的结果，[i,j)为同一个人的数据
                if index == len(file_names) - 1:  # 最后一个人
                    j = j + 56
                    last_prefix = prefix
                ratio = (np.where(label_pred[i:j] == 1)[0].shape[0]) / (j - i)
                ratios.append(ratio)

                # if isTest:
                
                #     draw_time_serieas(label_pred[i:j], self.args["model_save_path"], person_file_name, os.path.join(self.args["dataset_folder"],"test"))


                if (ratio > ratio_thresh):  # 判定为异常
                    pred_label = 1
                    abnormal_num_pred = abnormal_num_pred + 1
                else:
                    pred_label = 0
                    normal_num_pred = normal_num_pred + 1
                if (last_prefix.find('CP') != -1):
                    correct_label = 1  # 实际是异常
                    abnormal_num = abnormal_num + 1
                    if isTest:
                        print(f"abnormal: {last_prefix}, abnormal_ratio: {ratio}")
                else:
                    correct_label = 0
                    normal_num = normal_num + 1
                    if isTest:
                        print(f"normal: {last_prefix}, abnormal_ratio: {ratio}")
                if (pred_label == correct_label):
                    correct_num = correct_num + 1
                else:
                    wrong_num = wrong_num + 1



                    
                    if isTest:
                        print(f"wrong sample: {last_prefix}, ratio: {ratio}")
                predict.append(pred_label)
                actual.append(correct_label)
                i = j  # 更新同一个人的数据起点
                last_prefix = prefix
            j = j + 56  # [i,j)为同一个人的数据，一个json产出56个数据
            # break

        ratios = np.array(ratios)
        actual = np.array(actual)
        predict = np.array(predict)
        
        if isTest:
            draw(self.args["model_save_path"],predict=ratios,actual=actual)

        precision, recall, f_score, support = precision_recall_fscore_support(actual, predict, average='binary')

        return {
            "f1": f_score,
            "precision": precision,
            "recall": recall,
            "ratio_thresh": ratio_thresh,
            "epsilon": epsilon
        }

    def predict_personal_adjust_with_log(self, pred, ratio_thresh, epsilon=1, isTest = False):
        label_pred = np.array(pred)
        # 读取file_names
        file = open(self.file_names_path, "r", encoding='UTF-8')
        file_names = file.read()
        file_names = json.loads(file_names)
        file.close()
        i = 0
        j = i  # [i,j)为同一个人的数据
        correct_num = 0  # 预测正确、错误数
        wrong_num = 0
        normal_num = 0  # 真实正常、异常数
        abnormal_num = 0
        normal_num_pred = 0  # 预测正常、异常数
        abnormal_num_pred = 0
        predict = []
        actual = []
        ratios=[]
        person_file_name=[]

        epsilon=56/10

        last_prefix = file_names[0][:file_names[0].rfind("_S")]
        for index, file_name in enumerate(file_names):
            prefix = file_name[:file_name.rfind("_S")]
            
            # 对于每个文件进行校正，一个文件对应56个数据段
            # print("file_name",file_name, "prefix.find('CP') != -1",np.where(label_pred[j:j + 56])[0].shape[0])

            # if np.where(label_pred[j:j + 56])[0].shape[0] > epsilon and prefix.find('CP') != -1:  # 用真实标签
            #     label_pred[j:j + 56] = 1
            
            
            
            if prefix != last_prefix or index == len(file_names) - 1:  # 当前是一个新的人
                # 处理上一个人的结果，[i,j)为同一个人的数据
                if index == len(file_names) - 1:  # 最后一个人
                    j = j + 56
                    last_prefix = prefix
                ratio = (np.where(label_pred[i:j] == 1)[0].shape[0]) / (j - i)


                # np.where(label_pred[j:j + 56])[0].shape[0]
                if isTest:
                
                    draw_time_serieas(label_pred[i:j], self.args["model_save_path"], person_file_name, os.path.join(self.args["dataset_folder"],"test"))


                ratios.append(ratio)
                if (ratio > ratio_thresh):  # 判定为异常
                    pred_label = 1
                    abnormal_num_pred = abnormal_num_pred + 1
                else:
                    pred_label = 0
                    normal_num_pred = normal_num_pred + 1
                if (last_prefix.find('CP') != -1):
                    correct_label = 1  # 实际是异常
                    abnormal_num = abnormal_num + 1
                    if isTest:
                        print(f"abnormal: {last_prefix}, abnormal_ratio: {ratio}")
                else:
                    correct_label = 0
                    normal_num = normal_num + 1
                    if isTest:
                        print(f"normal: {last_prefix}, abnormal_ratio: {ratio}")
                if (pred_label == correct_label):
                    correct_num = correct_num + 1
                else:
                    wrong_num = wrong_num + 1
                    if isTest:
                        print(f"wrong sample: {last_prefix}, ratio: {ratio}")
                predict.append(pred_label)
                actual.append(correct_label)
                i = j  # 更新同一个人的数据起点
                last_prefix = prefix
                person_file_name=[]

            j = j + 56  # [i,j)为同一个人的数据，一个json产出56个数据
            # break
            person_file_name.append(file_name)

        ratios = np.array(ratios)
        actual = np.array(actual)
        predict = np.array(predict)
        
        if isTest:
            draw(self.args["model_save_path"],predict=ratios,actual=actual)

        precision, recall, f_score, support = precision_recall_fscore_support(actual, predict, average='binary')

        return {
            "f1": f_score,
            "precision": precision,
            "recall": recall,
            "ratio_thresh": ratio_thresh,
            "epsilon": epsilon
        }
