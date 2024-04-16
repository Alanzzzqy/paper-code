import torch
import torch.utils.data as Data
import numpy as np
import random



# def task_3x_to_x_seq(args, seq, dataset):
#     prompt = 3

#     # 随机选择一个位置
#     pos = random.randint(0, args.seq_len-2)

#     # 在选择的位置插入提示（prompt）为 3x 的元素
#     seq[pos] = prompt

#     # 从数据集中随机选择一个值 x，并将其插入到 seq 中的下一个位置
#     x = random.choice(dataset) + pos
#     seq[pos+1] = x

#     # 设置 seq 的末尾为 x，与提示值相同
#     seq[-1] = x

#     return seq

# 其实是x3_to_x，但是改函数名太麻烦，所以还叫这个
def task_3x_to_x_seq(args, seq, dataset):
    prompt = 3

    # 随机选择一个位置，在prompt之前
    pos = random.randint(1, args.seq_len - 1)

    # 在选择的位置插入提示（prompt）为 3x 的元素
    seq[pos] = prompt

    # 从数据集中随机选择一个值 x，并将其插入到 seq 中的上一个位置
    x = random.choice(dataset) + pos
    seq[pos - 1] = x

    # 设置 seq 的末尾为 x，与提示值相同
    seq[-1] = x

    return seq