import yaml
import argparse
from collections import defaultdict
import numpy as np
import csv
import os
import math
'''
Collect the results according to ./configs/sweep_configs/CIA/base.yaml, pick the last available result.

Usage:
different edge mask lr: please modify ./configs/GOOD_configs/base.yaml train.edge_mask_GNN_lr
'''

parser = argparse.ArgumentParser(description="")
parser.add_argument("--algorithm", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str, help="Dataset split. For example, degree (of Cora); university of WebKB")
parser.add_argument("--shift", type=str)
parser.add_argument("--model", type=str)
args = parser.parse_args()

config_path='./configs/GOOD_configs/base.yaml'
with open(config_path, 'r') as file:
    data = yaml.safe_load(file)
mask_lr_str=''
mask_lr_save_str=''
if data['use_inv_edge_mask']:
    #mask_lr=data['train']['edge_mask_GNN_lr'][0]
    if args.model=='CIAGAT':
        mask_lr=data['train']['edge_mask_GNN_lr']
    elif args.model in ['CIAGCN','GCN']:
        mask_lr=data['train']['GCN_edge_mask_GNN_lr']
    #mask_lr_str=f'_[{mask_lr}]mask-lr'
    mask_lr_str=f'_{mask_lr}mask-lr'
    mask_lr_save_str=f'edge_mask_lr-{mask_lr}'

dataset_config_path='./configs/sweep_configs/'+args.dataset+'.yaml'
with open(dataset_config_path, 'r') as file:
    data = yaml.safe_load(file)
lr=float(data['lr'][0])
#lr=0.01

bits_fn=lambda f: -int(math.floor(math.log10(abs(f))))
lr_bits=bits_fn(lr)

hyper_params_path = './configs/sweep_configs/CIA/base.yaml'
with open(hyper_params_path, 'r') as file:
    data = yaml.safe_load(file)
hyper_params=data['extra_param']


def get_test_acc(path):
    '''
    Get test acc from saved running logs.
    '''
    print(path)
    if os.path.exists(path):
        with open(path, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1]
            words = last_line.split()
            if words[-2]=='Loading':
                return -1
            if words[-2]=='Epoch': # 还没跑完
                test_acc=-1
                return test_acc
            test_acc = float(words[-2])  # 将字符串转换为浮点数
    else:
        test_acc=-1
    return test_acc



if args.algorithm=='CIA':
    if hyper_params[10][0]==1: # CIA
        save_dir=os.path.join('./storage/new_results',args.dataset, args.split, args.shift, args.algorithm, args.model)
        os.makedirs(save_dir, exist_ok=True)
        lbda=hyper_params[0]
        suffix=''
        for i in range(1,15):
            suffix+=f'_{hyper_params[i][0]}'
            
        res_mean={}
        res_std={}
        

        for l in lbda: # lambdas
            accs=[]
            for rd in range(1,4): # round 1~3
                #print(rd)
                res_log_path=f'./storage/log/round{rd}/'+args.dataset+'_'+args.split+'_'+args.shift+'/'+args.model \
                +f'_3l_meanpool_0.5dp/{lr:.{lr_bits}f}lr'+'_0.0wd/'+args.algorithm+'_0.1'+f'_{l}'+suffix+'/lb_sweeping.log'
                accs.append(get_test_acc(res_log_path))
            res_mean[l]=np.mean(accs)
            res_std[l]=np.std(accs)
        
        filename=os.path.join(save_dir, 'result.txt')
        with open(filename, 'w') as file:
            for l in lbda:
                print(f"{l:<16}", end='')
                file.write(f'{l} ')
            print()
            file.write('\n')
            max_mean = float('-inf')
            max_l = None
            
            for l in lbda:
                mean = res_mean[l]
                std = res_std[l]
                if mean > max_mean:
                    max_mean = mean
                    max_l = l
                print(f'{f"{mean*100:.2f}({std*100:.2f})":<16}', end='')
                file.write(f"{mean*100:.2f}({std*100:.2f}) ")
            print()
            file.write('\n')
            print(f"Best: {max_mean*100:.2f}, lambda={max_l}")
    elif hyper_params[7][0]==1: # CIA-LRA
        save_dir=os.path.join('./storage/new_results',args.dataset, args.split, args.shift, args.algorithm, args.model, mask_lr_save_str)
        os.makedirs(save_dir, exist_ok=True)
        hops=hyper_params[8]
        lbda=hyper_params[9]
        prefix=''
        for i in range(8):
            prefix+=f'_{hyper_params[i][0]}'
        suffix=''
        for i in range(10,15):
            suffix+=f'_{hyper_params[i][0]}'
            
        res_mean=defaultdict(dict)
        res_std=defaultdict(dict)
        
        for h in hops: # hops
            for l in lbda: # lambdas
                accs=[]
                for rd in range(1,3): # round 1~3
                    res_log_path=f'./storage/log/round{rd}/'+args.dataset+'_'+args.split+'_'+args.shift+'/'+args.model \
                    +f'_3l_meanpool_0.5dp/{lr:.{lr_bits}f}lr'+mask_lr_str+'_0.0wd/'+args.algorithm+'_0.1'+prefix+f'_{h}_{l}'+suffix+'/lb_sweeping.log'
                    accs.append(get_test_acc(res_log_path))
                res_mean[h][l]=np.mean(accs)
                res_std[h][l]=np.std(accs)
            
        filename=os.path.join(save_dir, 'result.txt')
        with open(filename, 'w') as file:
            print(" "*5, end='')
            file.write(' ')
            for l in lbda:
                print(f"{l:<16}", end='')
                file.write(f'{l} ')
            print()
            file.write('\n')
            max_mean = float('-inf')
            max_h = None
            max_l = None
            
            for h in hops:
                print(f'{h:<5}', end='')
                file.write(f'{h} ')
                for l in lbda:
                    mean = res_mean[h][l]
                    std = res_std[h][l]
                    if mean > max_mean:
                        max_mean = mean
                        max_h = h
                        max_l = l
                    print(f'{f"{mean*100:.2f}({std*100:.2f})":<16}', end='')
                    file.write(f"{mean*100:.2f}({std*100:.2f}) ")
                print()
                file.write('\n')
            print(f"Best: {max_mean*100:.2f}, hop={max_h}, lambda={max_l}")
        



