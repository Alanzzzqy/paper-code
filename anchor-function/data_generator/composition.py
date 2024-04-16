import torch
import torch.utils.data as Data
import numpy as np
import random


def single_func(x, single_prompt):
        p_list = [1,2,3,4]
        diff = [5, 1, -2, -8]
        # diff = [5, 1, -2, -5]
        i = p_list.index(single_prompt)
        return x + diff[i]


def composition_seq(args, seq, dataset, prompt_first=True):
    r'''
        使用两个简单的prompt复合成一个复杂的prompt
        扣掉34x这个case，来研究模型的泛化性
        prompt_first: True表示prompt在前，False表示prompt在后
    '''
    p1 = random.choice([1, 2, 3, 4])
    if p1 == 3:
        p2 = random.choice([1, 2, 3])
    else:
        p2 = random.choice([1, 2, 3, 4])
    
    # 随机选取一个位置，将该位置的数替换成p1，下一位替换成p2
    pos = random.randint(0, args.seq_len-3) # randint是包含最后一个数字的
    x = random.choice(dataset) + pos
    if prompt_first:
        seq[pos], seq[pos+1], seq[pos+2] = p1, p2, x
    else:
        seq[pos], seq[pos+1], seq[pos+2] = x, p1, p2
    
    tmp = single_func(x, p2)
    y = single_func(tmp, p1)
    seq[-1] = y

    return seq


def composition_seq_specific(args, seq, dataset, mode = '11', prompt_first=True):
    if mode == '11':
        p1, p2 = 1, 1
    elif mode == '12':
        p1, p2 = 1, 2
    elif mode == '13':
        p1, p2 = 1, 3
    elif mode == '14':
        p1, p2 = 1, 4
    elif mode == '21':
        p1, p2 = 2, 1
    elif mode == '22':
        p1, p2 = 2, 2
    elif mode == '23':
        p1, p2 = 2, 3
    elif mode == '24':
        p1, p2 = 2, 4
    elif mode == '31':
        p1, p2 = 3, 1
    elif mode == '32':
        p1, p2 = 3, 2
    elif mode == '33':
        p1, p2 = 3, 3
    elif mode == '34':
        p1, p2 = 3, 4
    elif mode == '41':
        p1, p2 = 4, 1
    elif mode == '42':
        p1, p2 = 4, 2
    elif mode == '43':
        p1, p2 = 4, 3
    elif mode == '44':
        p1, p2 = 4, 4

    # 随机选取一个位置，将该位置的数替换成p1，下一位替换成p2
    pos = random.randint(0, args.seq_len-3) # randint是包含最后一个数字的
    x = random.choice(dataset) + pos
    if prompt_first:
        seq[pos], seq[pos+1], seq[pos+2] = p1, p2, x
    else:
        seq[pos], seq[pos+1], seq[pos+2] = x, p1, p2
    
    tmp = single_func(x, p2)
    y = single_func(tmp, p1)
    seq[-1] = y

    return seq