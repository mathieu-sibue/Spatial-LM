
import torch

import os
import argparse
from torch.utils.data import DataLoader

import torch
import pickle
from utils.params import Params
# from torch_geometric.transforms import NormalizeFeatures
import mydataset
from LMs import trainer
import LMs

def parse_args(config_path):
    parser = argparse.ArgumentParser(description='run the model')
    parser.add_argument('--config', dest='config_file', default = config_path)
    return parser.parse_args()

if __name__=='__main__':

    # Section 1, parse parameters
    args = parse_args('config/train.ini') # from config file
    params = Params()   # put to param object
    params.parse_config(args.config_file)
    params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', params.device)

    # section 2, objective function and output dim/ move to trainer
    # this is usually covered by huggingface models
    mydata = mydataset.setup(params)

    # section 3, get the model
    model = LMs.setup(params)

    # section 5, train (finetune)
    HFTrainer.train(params, model, mydata)


    