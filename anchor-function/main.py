import numpy as np
import torch

from model import *
import argparse
import os
import shutil
from data import *
from train import *

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader 


def main(args):
    # 设置随机种子
    setup_seed(args.seed)

    for file in ['pic', 'loss', 'src', 'data', 'model']:
        os.makedirs(f'{args.working_dir}/{file}', exist_ok=True)
    datas = get_data(args, return_dict=True)

    train(args, datas)

    # train_data_loader, test_data_loader, train_seq_group, test_seq_group, train_seq_list, test_seq_list
    # train(args, train_data_loader, test_data_loader, train_seq_group, test_seq_group)
    # train(args, train_data_loader, test_data_loader, train_seq_list, test_seq_list)


# 在这里调整训练的各种参数
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pytorch distributed")

    # 数据集参数
    parser.add_argument('-N_train', '--train_data_size', type = int, default = 1000) 
    parser.add_argument('-N_test', '--test_data_size', type = int, default = 500) 
    parser.add_argument('-sl', '--seq_len', type = int, default = 9)
    parser.add_argument('-dmin', '--data_min', type = int, default = 20)
    parser.add_argument('-dmax', '--data_max', type = int, default = 100)

    parser.add_argument('-dmode', '--data_mode', nargs='*', type=str, default = [1], help='各类数据集的模式，不同任务中的数据集模式不同')
    parser.add_argument('-dp', '--data_percent', nargs='*', type=float, default = [1], help='各类数据集占比')
    parser.add_argument('-dn', '--data_name', nargs='*', type=str, default = ['full data'], help='各类数据集名称')
    parser.add_argument('-dmask', '--data_mask', nargs='*', type=int, default = [0], help='是否mask该类数据集，1表示mask，0表示不mask，mask后的数据集不参与训练')
    parser.add_argument('-dshow', '--data_show', nargs='*', type=int, default = [0], help='画图时是否显示该类数据集，1表示显示，0表示不显示')

    # 目标函数
    parser.add_argument('-func', '--target', type = str, default = '3x_to_x')

    
    # 网络超参数
    parser.add_argument('-bs', '--batch_size', type = int, default = 10) 
    parser.add_argument('-vs', '--vocab_size', type = int, default = 201) 
    parser.add_argument('-mp', '--max_pos', type = int, default = 20)
    parser.add_argument('-dm', '--d_model', type = int, default = 400)
    parser.add_argument('-d_ff', '--d_feedforward', type = int, default = 1200)
    parser.add_argument('-dk', '--d_k', type = int, default = 64)
    parser.add_argument('-dv', '--d_v', type = int, default = 64)
    parser.add_argument('-nl', '--n_layers', type = int, default = 4)
    parser.add_argument('-nh', '--n_heads', type = int, default = 4)
    parser.add_argument('-cl', '--clip', type = int, default = 1)

    parser.add_argument('-ne', '--n_epoch', type = int, default = 3000) 
    # parser.add_argument('--schedu', choices = ['StepLR'], default = 'StepLR') 
    parser.add_argument('-lr', '--lr', type = float, default = 1.e-4) 
    parser.add_argument('-lds', '--lr_decay_step', type = int, default = 1000) 
    parser.add_argument('-ldr', '--lr_decay_rate', type = float, default = 1)  
    parser.add_argument('-seed', '--seed', type = int, default = 1)  
    parser.add_argument('-scheduler', '--scheduler', type = str, default = 'GradualWarmupScheduler_CosineAnnealingLR')

    parser.add_argument('-device','--device_rank', default='0',help='CUDA rank')
    # 网络结构
    parser.add_argument('-m', '--model', type = str, default = 'GPT') 
    parser.add_argument('-op', '--optim', choices = ['Adam', 'SGD', 'AdamW'], default = 'AdamW')  


    # 保存、输出信息和画图的间隔
    parser.add_argument('-sme', '--save_model_epoch', type = int, default = 500)
    parser.add_argument('-ple', '--print_loss_epoch', type = int, default = 10)
    parser.add_argument('-pae', '--print_acc_epoch', type = int, default = 100)
    parser.add_argument('-plae', '--plot_loss_acc_epoch', type = int, default = 500)
    
    # 前缀与后缀
    parser.add_argument('-prefix', '--prefix', type = str, default = ' ')
    parser.add_argument('-suffix', '--suffix', type = str, default = ' ')

    # 大文件夹的后缀
    parser.add_argument('-dir_suffix', '--dir_suffix', type = str, default = ' ')

    # 实验编号与实验名称
    parser.add_argument('-ei', '--exp_index', type = str, default = '0')
    parser.add_argument('-en','--exp_name',type = str , default= '')
    parser.add_argument('-yon','--yes_or_no',type = str , default= '')

    args = parser.parse_args()

    # 生成主文件夹目录
    # working_dir = f'{args.target}-seed_{int(args.seed)}-N_{int(args.train_data_size)}'
    # working_dir = f'{args.target}-seed_{int(args.seed)}-N_{int(args.train_data_size)}-layers_{int(args.n_layers)}
    working_dir = f'{int(args.seed)}-{int(args.train_data_size)}-{int(args.n_layers)}-{args.yes_or_no}-{int(args.n_heads)}'
    if args.prefix != ' ':
        working_dir = f'{args.prefix}-{working_dir}'
    if args.suffix != ' ':
        working_dir = f'{working_dir}-{args.suffix}'
    
    if args.dir_suffix != ' ':
        args.working_dir = f'./result/GPT_{args.dir_suffix}/{args.exp_name}/{args.exp_index}/{working_dir}/'
    else:
        args.working_dir = f'./result/{args.model}/{working_dir}/'

    main(args)