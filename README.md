# AT-IG
Codes of "Detect early neurodevelopmental impairment for infants via time-series anomaly detection"

## Environment setting
    Python=3.8
    Pytorh=1.7.0 (py3.8_cuda11.0.221)
    torchvision=0.8.1  

## Dataset setting
 1. The used skeleton datasets can be requested by email. 
                            

## Training and test
      python main_bone.py --input_c 150 --output_c 150  --model_save_path "checkpoints/attention_visalization" --seed seed_2021 --dataset_folder [data_path]/seed_2021 --data_path [data_path]/seed_2021/precessed --gpu_id 0  --add_info_gain True


      
        
