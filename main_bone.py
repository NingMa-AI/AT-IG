
from args import get_parser
from torch.backends import cudnn
from utils.utils import *

import matplotlib.pyplot as plt
from solver_bone import Solver

import random
import json
from datetime import datetime
import csv

def str2bool(v):
    return v.lower() in ('true')

def plot_losses(solver, save_path, plot=False):
    plt.plot(solver.info_gains, label='Information-Gain')
    plt.plot(solver.recon_losses, label='Reconstruction-Error')
    plt.plot(np.array(solver.train_loss1)/20.0, label='Association-Discrepancy')
    plt.title("The losses during training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    with open(os.path.join(save_path,'loss-ablation.csv'), 'w') as f:
        writer=csv.writer(f)
        writer.writerow(["Iteration","Information-Gain", "Reconstruction-Error","Association-Discrepancy"])
        for (i, (a,b,c))in enumerate(zip(solver.info_gains, solver.recon_losses, solver.train_loss1)):
            writer.writerow([i, a, b, c/20.0])

    # np.save(os.path.join(save_path,"Information-Gain.npy"),np.array(solver.info_gains))
    # np.save(os.path.join(save_path,"Reconstruction-Error.npy"), np.array(solver.recon_losses))
    # np.save(os.path.join(save_path,"Association-Discrepancy.npy"), np.array(solver.train_loss1)/20.0)

    # plt.plot(solver.train_loss1, label='AssDis')
    # # plt.plot(solver.train_loss2, label='Prior_train')
    # plt.title("Series-Prior losses during training")
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(f"{save_path}/Series-Prior-losses.png", bbox_inches="tight")
    # if plot:
    #     plt.show()
    # plt.close()

    # plt.plot(solver.valid_loss1, label='Series_val')
    # plt.plot(solver.valid_loss2, label='Prior_val')
    # plt.title("Validation losses during training")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    # if plot:
    #     plt.show()
    # plt.close()

def train(config):

    out_path = config.data_path
    config.file_names_path = out_path + '_file_names'
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    # 训练
    # 1.正常跑训练
    checkpoint_id = solver.train()
    solver.checkpoint_id = checkpoint_id
    # save_path = './output/'+checkpoint_id.split('.')[0]+'/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # 验证
    re = solver.test()
    plot_losses(solver,config.model_save_path)
    return checkpoint_id, re

def test(config, checkpoint_id, thresh, ratio_thresh,epsilon):
    out_path = config.data_path + '_test'
    config.data_path = out_path
    config.file_names_path = out_path + '_file_names'
    config.checkpoint_id = checkpoint_id
    solver_test = Solver(vars(config))
    re = solver_test.test(thresh= thresh, ratio_thresh= ratio_thresh, isTest= True, epsilon= epsilon)

    return re

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # np.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(config):
    result = {}
    # 1. 训练
    # config.dataset_root_paths = config.dataset_folder + '/train'
    checkpoint_id, re = train(config)
    result["train_result"] = re
    result["checkpoint_id"] = checkpoint_id

    # 2. 测试
    # config.dataset_root_paths = config.dataset_folder + '/test'
    re = test(config, checkpoint_id, re['thresh'], re['ratio_thresh'],re["epsilon"])
    result["test_result"] = re
    return result


if __name__ == '__main__':
    import warnings,sys

    warnings.filterwarnings('ignore')

    id = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    torch.autograd.set_detect_anomaly(True)
    parser = get_parser()

    config = parser.parse_args()
    setup_seed()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu_id)
    
    config.model_save_path=os.path.join(config.model_save_path, "time_"+id+"_seed_"+str(config.seed)+"_info_"+str(config.add_info_gain))
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    # sys.stdout = open(os.path.join(config.model_save_path,'output.txt'),'w')

    # Save config
    args_path = f"{config.model_save_path}/config.txt"
    with open(args_path, "w") as f:
        # args.epsilon=epsilon
        # args.threshold=best_re["threshold"]
        # args.model_id=id
        json.dump(config.__dict__, f, indent=2)


    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)

