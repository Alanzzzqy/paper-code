import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import shutil
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import textwrap
sys.path.append('/content/drive/MyDrive/anchor-function/anchor-function') 
from model import *
from utils import *
from data import *
sys.path.remove('/content/drive/MyDrive/anchor-function/anchor-function')

# 参数设置区
n_epoch = 1000
exp_name = "forward" #实验名称
# 这里的测试句子用的是seed1 测试集中的第一句
input = torch.tensor([37, 3, 81, 52, 35, 83, 77, 80, 68])
input = input.reshape(1, 9)
# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = input.to(device)

# 标准模型的绝对路径
std_file_absolute_path = '/content/drive/MyDrive/anchor-function/anchor-function/result/GPT_3x_to_x_for_paper/norm_exp'



exp_index = '4'
N_seed = 10
N_train = 1000
n_layers = 2
n_heads = 1

def load_model(model_path, config_path,yes_or_no):
    """
    加载模型和配置文件。

    Parameters:
    - model_path (str): 模型文件路径，包含模型文件夹和文件名。
    - config_path (str): 参数文件路径，包含配置文件夹和文件名。

    Returns:
    - model (myGPT): 加载的模型。
    - args (argparse.Namespace): 加载的配置参数。
    """
    
    # 加载模型参数
    model_state_dict = torch.load(f'{model_path}/model/model_{n_epoch-1}.pt', map_location=device)

    # 加载配置参数
    args = read_json_data(f'{config_path}/config.json')
    args = argparse.Namespace(**args)
    # print(args)

    # 创建模型并加载参数
    if yes_or_no =="yes": model = myGPT(args, device)
    elif yes_or_no =="no": model = myGPT_without(args,device)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model, args

def plot_attention_heatmaps(dec_self_attns, input, seed , n_epoch, yes_or_no):
    """
    绘制注意力热力图。

    Parameters:
    - dec_self_attns (torch.Tensor): 注意力张量，形状为 [num_layers, num_heads, sequence_length, sequence_length]。
    - input (torch.Tensor): 输入张量。
    - seed (int): 种子
    - n_epoch (int): 训练周期。
    - yes_or_no (str): 是否含有某些模块。
    """
    # 定义图形的大小和子图的布局
    fig, axes = plt.subplots(nrows=dec_self_attns.shape[0], ncols=dec_self_attns.shape[1],
                             figsize=(4*dec_self_attns.shape[1], 4*dec_self_attns.shape[0]))
    # 创建子图数组，保持二维
    if dec_self_attns.shape[1] == 1:
      axes = axes.reshape((dec_self_attns.shape[0],1))

    # 循环遍历每个头，每个层，并在子图上绘制热力图
    for layer in range(dec_self_attns.shape[0]):
        for head in range(dec_self_attns.shape[1]):
            # 提取对应头的注意力权重矩阵
            attention_matrix = dec_self_attns[layer, head, :, :].cpu().detach().numpy()
            # 绘制热力图
            if head == dec_self_attns.shape[1]-1: char_value = True
            else: char_value = False
            sns.heatmap(attention_matrix, cmap="YlGnBu", ax=axes[layer, head], annot=False, cbar=char_value, fmt=".1f")
            # sns.heatmap(attention_matrix, cmap="YlGnBu", ax=axes[layer, head], annot=annot_values, cbar=char_value, fmt="")
            # 设置子图标题
            axes[layer, head].set_title(f'Layer {layer + 1}, Head {head + 1}')

    # 设置整个图的标题
    pic_title = f"input:{str(input[0].tolist())} seed{seed}  ne:{n_epoch}  {exp_name}:{yes_or_no}"
    # 单头时标题无法显示启用下面这行语句
    pic_title = '\n'.join(textwrap.wrap(pic_title, width=40))
    plt.suptitle(pic_title, fontsize=16)
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join('attn_pic', f"seed{seed}_ne{n_epoch}_{yes_or_no}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

if not os.path.exists('attn_pic'):
    os.makedirs('attn_pic')



for i in range(N_seed):
    std_file = str(i + 1) + '-' + str(N_train) + '-' + str(n_layers) + "-yes-" + str(n_heads)
    # std_file = os.path.join(std_file_absolute_path, exp_index, std_file)
    exp_file = str(i + 1) + '-' + str(N_train) + '-' + str(n_layers) + "-no-" + str(n_heads)
    
    # 创建模型并加载参数
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_std, _ = load_model(std_file,std_file,"yes")
    model_exp, _ = load_model(exp_file,exp_file,"no")
    model_exp.to(device)
    

    # 标准模型
    output , dec_self_attns_std = model_std.forward(input) # output是每个词之后出现的那个词的概率，vocab一共是201
    # dec_self_attns 是一个list，将它堆叠成一个张量
    dec_self_attns_std = torch.cat(dec_self_attns_std, dim=0)
    plot_attention_heatmaps(dec_self_attns_std,input,i+1 ,n_epoch,"yes")
    
    # print(model_exp.device)
    # print(input.device)

    # 实验模型
    output , dec_self_attns_exp = model_exp.forward(input) # output是每个词之后出现的那个词的概率，vocab一共是201
    # dec_self_attns 是一个list，将它堆叠成一个张量
    dec_self_attns_exp = torch.cat(dec_self_attns_exp, dim=0)
    plot_attention_heatmaps(dec_self_attns_exp,input, i+1 ,n_epoch-1,"no")


