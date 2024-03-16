import argparse
import os
import shutil
import logging
import yaml
import pandas as pd 
import numpy as np
from get_data_deep import get_data_deep 

def train_and_test_deep(config_file):
    config = get_data_deep(config_file)
    root_dir = config['data_source']['data_src']
    dest = config['load_data_deep']['preprocessed_data']
    p = config['load_data_deep']['full_path']
    cla = config['data_source']['data_src']
    cla = os.listdir(cla)
    
    splitr =config['train_split']['split_ratio']
    for k in range(len(cla)):
        per = len(os.listdir((os.path.join(root_dir,cla[k]))))
        print(k,"->",per)  
        cnt = 0
        split_ratio = round((splitr/100)*per)
        
        for j in os.listdir((os.path.join(root_dir,cla[k]))):
            pat = os.path.join(root_dir+'/'+cla[k],j)
           #print(pat)
            if(cnt!=split_ratio):
                shutil.copy(pat, dest+'/'+'train/class_'+str(k))
                cnt+=1
            else:
                shutil.copy(pat, dest+'/'+'train/class_'+str(k))
        print('done')
        
        

if __name__=="__main__":
    args = argparse.ArguementParser()
    args.add_arguement("--config",default="deep_params.yaml")
    parsed_args=args.parse_args()
    data = get_data_deep(config_file=parsed_args.config)