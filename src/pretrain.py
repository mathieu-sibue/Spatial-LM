
import torch

import os
import argparse
from torch.utils.data import DataLoader

import torch
import pickle
from utils.params import Params
# from torch_geometric.transforms import NormalizeFeatures
import pretrain_dataset
from LMs import trainer
import LMs

def parse_args(config_path):
    parser = argparse.ArgumentParser(description='run the model')
    parser.add_argument('--config', dest='config_file', default = config_path)
    return parser.parse_args()
    
if __name__=='__main__':

    # Section 1, parse parameters
    args = parse_args('config/pretrain.ini') # from config file
    params = Params()   # put to param object
    params.parse_config(args.config_file)
    params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', params.device)

    # section 2, objective function and output dim/ move to trainer
    # this is usually covered by huggingface models

    # section 3, model, loss function, and optimizer
    # load from cs model checkpoint
    model = LMs.setup(params).to(params.device)

    # section 4, saving path for output model
    params.dir_path = trainer.create_save_dir(params)    # prepare dir for saving best models, put config info first

    # section 4, load data; prepare output_dim/num_labels, id2label, label2id for section3; 
    # 4.1 traditional train
    # mydata = pretrain_dataset.setup(params)
    # print(mydata.masked_train_dataset)
    # best_f1 = trainer.train(params, model, mydata)

    # 4.2 train many datasets
    for i in range(3, 8):
        params.rvl_cdip = '/home/ubuntu/air/vrdu/datasets/rvl_pretrain_datasets/'+str(i)+'_fixed_bert.hf'
        mydata = pretrain_dataset.setup(params)
        best_f1 = trainer.pretrain(params, model, mydata)
        del mydata


    # section 5, inference only (on test_dataset)
    # inferencer.inference(params,model,mydata,'v3_base_benchmark_Jan_1pm.json')


