import torch
import torch.utils.data as Data
import numpy as np
import random



def multitask_seq(args, seq, dataset):
    r'''
        10个任务一起做
    '''
    prompt = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    pos = random.randint(0, args.seq_len-2) # randint是包含最后一个数字的
    seq[pos] = prompt
    x = random.choice(dataset) + pos
    def single_function(x, prompt):
        p_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        diff = [-4, 3, -5, -2, 1, 4, -3, -1, 2, 5]
        i = p_list.index(prompt)
        return x + diff[i]
    seq[-1] = single_function(x, prompt)

    return seq