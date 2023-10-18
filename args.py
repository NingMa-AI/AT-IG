import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=3)    # 修改
    parser.add_argument('--k', type=int, default=0.01) # yanxueya 0.01
    parser.add_argument('--input_c', type=int, default=75)  # 修改：输入特征数
    parser.add_argument('--output_c', type=int, default=75)  # 修改：输出特征数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='BONE')
    parser.add_argument('--gpu_id', type=str, default='5')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')   # 模型存放位置
    parser.add_argument('--win_size', type=int, default=20)  # x窗口长度
    parser.add_argument('--predict_win', type=int, default=5)  # y窗口长度
    parser.add_argument('--step', type=int, default=5)      # 划分数据段的步长

    parser.add_argument("--dataset_folder", type=str,
                        default="/data/yanxueya/data/new/anomaly-detection/CP-NO-all/rotation_clean_all_binjiang_center_joint/seed_2021")  # 原始json文件存放的总目录，该目录下有train和test两个文件夹
    parser.add_argument('--data_path', type=str, default='./dataset/processed7') # 处理好的数据文件夹
    parser.add_argument('--seed', type=int, default=2020)       # seed
    parser.add_argument('--add_info_gain', type=str2bool, default=True)  # 是否加信息增益
    parser.add_argument("--abnormal_ratio", type=float, default=0.0)       # 婴儿异常预测阈值
    parser.add_argument("--abnormal_label", type=str, default="CP")
    parser.add_argument("--model_folder", type=str, default="")       # 原始json文件存放的总目录，该目录下有train和test两个文件夹
    parser.add_argument("--output_folder", type=str, default="datasets/bone/processed5")  
    parser.add_argument("--group", type=str, default="1-1", help="Required for SMD dataset. <group_index>-<index>")
    parser.add_argument("--lookback", type=int, default=20)             # test x窗口长度
    parser.add_argument("--normalize", type=str2bool, default=False)    # test
    parser.add_argument("--spec_res", type=str2bool, default=False)
    parser.add_argument("--time_vis", type=str2bool, default=False)
    return parser