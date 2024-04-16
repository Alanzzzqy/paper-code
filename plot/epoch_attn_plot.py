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
sys.path.append('/content/drive/MyDrive/anchor-function/anchor-function') 
from model import *
from utils import *
from data import *
sys.path.remove('/content/drive/MyDrive/anchor-function/anchor-function')

# 参数设置区
exp_name = "norm" # 实验名称
n_epoch = 1000
f_name = "1-2000-2-no-4"

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 这里的测试句子用的是seed1 测试集中的第一句
input = torch.tensor([37, 3, 81, 52, 35, 83, 77, 80, 68])
input = input.reshape(1, 9)
input = input.to(device)

# 自动参数区
params_list = f_name.split('-')
seed = int(params_list[0])
N_train = int(params_list[1])
n_layers = int(params_list[2])
yes_or_no = params_list[3]
n_heads = int(params_list[4])
# print(seed,N_train,n_layers,yes_or_no,n_heads)




def load_model(model_path, config_path, yes_or_no, model_index):
    """
    加载模型和配置文件。

    Parameters:
    - model_path (str): 模型文件路径，包含模型文件夹和文件名。
    - config_path (str): 参数文件路径，包含配置文件夹和文件名。
    - yes_or_no (str): 是否含有某些模块。
    - model_index (int)： 模型的编号

    Returns:
    - model (myGPT): 加载的模型。
    - args (argparse.Namespace): 加载的配置参数。
    """
    
    # 加载模型参数
    model_state_dict = torch.load(f'{model_path}/model/model_{model_index}.pt', map_location=device)

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
    plt.suptitle(pic_title, fontsize=16)
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_pic_path, f"seed{seed}_ne{n_epoch}_{yes_or_no}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

save_pic_path = os.path.join(f_name, 'attn_pic')
if not os.path.exists(save_pic_path):
    os.makedirs(save_pic_path)

# 创建模型的编号列表
model_folder = f'{f_name}/model'  
model_files = os.listdir(model_folder)
# 提取轮次编号
model_index = [int(file.split('_')[-1].split('.')[0]) for file in model_files if file.startswith('model_') and file.endswith('.pt')]
# 打印轮次编号
print(model_index)


for i in model_index:
    # 创建模型并加载参数
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(f_name,f_name,yes_or_no,i)

    output, dec_self_attns = model.forward(input) # output是每个词之后出现的那个词的概率，vocab一共是201
    # dec_self_attns 是一个list，将它堆叠成一个张量
    dec_self_attns = torch.cat(dec_self_attns, dim=0)
    plot_attention_heatmaps(dec_self_attns, input, seed, i, yes_or_no)
