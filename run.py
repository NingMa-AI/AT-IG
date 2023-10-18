import os
import time,pynvml
def getAvaliableDevice(gpu=[4,5,2,3,1,0],min_mem=24000,left=False):
# def getAvaliableDevice(gpu=[6],min_mem=10000,left=False):
    """
    :param gpu:
    :param min_mem:
    :param left:
    :return:
    """
    # return 0
    pynvml.nvmlInit()
    t=int(time.strftime("%H", time.localtime()))

    if t>=23 or t <8:
        left=False # do not leave any GPUs
    #else:
        #left=True

    min_num=3
    dic = {0: 5, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4,-1: -1}  # just for 120 server
    ava_gpu = -1

    while ava_gpu == -1:
        avaliable=[]
        for i in gpu:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            # print((utilization.gpu))
            if (meminfo.free / 1024 ** 2)>min_mem and utilization.gpu<50:
                avaliable.append(dic[i])
            # elif i ==0 and (meminfo.free / 1024 ** 2)>16000:
            #     avaliable.append(dic[i])

            elif (meminfo.free / 1024 ** 2)>16000 and utilization.gpu<1:
                avaliable.append(dic[i])

        if len(avaliable)==0 or (left and len(avaliable)<=1):
            ava_gpu = -1
            time.sleep(30)
            continue
        ava_gpu= avaliable[0]
    return ava_gpu

dirs=[
    #  "/data/maning/dataset/CP_NCP_anomaly_detection/CP-NO-1-10/rotation_clean_all_binjiangcoordinate_angle_center_joint/",
    #  "/data/maning/dataset/CP_NCP_anomaly_detection/CP-NO-1-10/rotation_clean_all_hubincoordinate_angle_center_joint/",
    #  "/data/maning/dataset/CP_NCP_anomaly_detection/CP-NO-1-10/rotation_clean_binjiang_and_hubin_angle_center_joint/",
    #  "/data/maning/dataset/CP_NCP_anomaly_detection/CP-NO-1-10/hubin_to_binjiang/",
    #  "/data/maning/dataset/CP_NCP_anomaly_detection/CP-NO-1-10/binjiang_to_hubin/"
     # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-1-10/rotation_clean_all_binjiangcoordinate_angle_center_joint/"
    #  "/data/maning/dataset/CP_NCP_anomaly_detection/CP-NO-1-10/rotation_clean_all_binjiangcoordinate_angle_center_joint/"
    #  "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-all/rotation_clean_all_hubincoordinate_angle_center_joint/",
    #  "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-all/rotation_clean_all_hubincoordinate_angle_resolution/"
    # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-all/rotation_clean_all_binjiangcoordinate_angle_center_joint/",
    # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-all/rotation_clean_all_hubincoordinate_angle_center_joint/"
    #  "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-all/rotation_clean_all_binjiangcoordinate_angle_center_joint/"

     # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection-2/CP-NO-1-10/hubin_and_binjiang_resolution/",
     # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection-2/CP-NO-1-10/hubin_2_binjiang_resolution/",
     # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection-2/CP-NO-1-10/binjiang_2_hubin_resolution/",
     # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection-2/CP-NO-1-10/rotation_clean_all_binjiangcoordinate_angle_resolution/",
     # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection-2/CP-NO-1-10/rotation_clean_all_hubincoordinate_angle_resolution/",
     # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-1-10/clean_all_binjiangcoordinate_angle_resolution/",
     # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-1-10/rotation_clean_all_binjiangcoordinate_angle_resolution/",
     # "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-1-10/rotation_clean_all_hubincoordinate_angle_resolution/"
     "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-all/hubin_and_binjiang/",
    #  "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-all/binjiang_2_hubin/",
    #  "/data1/zhanghongyi/erbao/coordinate_angle/anomaly-detection/CP-NO-all/hubin_2_binjiang/",
     ]

log_folder="checkpoints/attention_visalization"
abnormal_ratio=0

for yydir in dirs:
    for seed in  ["seed_2021", "seed_2022", "seed_2023"]: #,"",  
        output_folder= os.path.join(yydir,seed,"precessed")   
        dataset_folder=os.path.join(yydir,seed)
        
        # if not os.path.exists(output_folder):
        # os.system("python "+ "preprocess.py" +" --abnormal_label CP --dataset_folder {} --output_folder {}  --abnormal_ratio {}"
        #     .format(dataset_folder, output_folder,abnormal_ratio))

        gpu_id=getAvaliableDevice(min_mem=22000,left=True)
        os.system("python "+ "main_bone.py" +"   --input_c {} --output_c {}  --model_save_path {}  --seed {} --dataset_folder {} --data_path {} --gpu_id {}  --add_info_gain True " #
                .format(150,150, log_folder, seed.split("_")[1], dataset_folder,output_folder,gpu_id)) 
        time.sleep(120)

        # gpu_id=getAvaliableDevice(min_mem=20000,left=True)
        # os.system("python "+ "main_bone.py" +"   --input_c {} --output_c {} --model_save_path {}  --seed {} --dataset_folder {} --data_path {} --gpu_id {}  --add_info_gain False &" #
        #         .format(150,150, log_folder, seed.split("_")[1], dataset_folder,output_folder,gpu_id)) 
        # time.sleep(60)
        # gpu_id=getAvaliableDevice(min_mem=24000)
        # os.system("python "+ "train_bone.py" +" --log {} --abnormal_label CP  --dataset_folder {} --output_folder {} --epoch 200 --gpu_id {} --add_info_gain True &" #
        #         .format(log_folder, dataset_folder,output_folder,gpu_id)) 
        # time.sleep(30)