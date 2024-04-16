import torch
import torch.utils.data as Data
import numpy as np
import random
import math
from data_generator import *

class MyDataSet(Data.Dataset):
    def __init__(self,datas):
        self.datas = datas

    def __getitem__(self, item):
        data = self.datas[item]
        decoder_input = data[:-1] # decoder_input 包含输入序列中的所有元素，但是末尾的元素被省略了。它表示解码器（decoder）的输入序列。
        decoder_output = data[1:] # decoder_output 包含与 decoder_input 相同的元素，但是开头的元素被省略了。它表示解码器应该生成的目标输出序列

        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)

        return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
                "decoder_output": decoder_output, "decoder_output_len": decoder_output_len}

    def __len__(self):
        return len(self.datas)

    def padding_batch(self, batch):
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)
        # batch是一个列表，其中每一个是一个字典，含有‘decoder_input’和‘decoder_output’两个键

        return decoder_inputs, decoder_outputs
    

def generate_random_list(seq_len=7, data_min=20, data_max=100):
    r'''生成给定长度的随机数列表，每个数的范围是[data_min, data_max]'''
    return [random.randint(data_min, data_max) for _ in range(seq_len)]


def generate_mod_list(data_min=20, data_max=100, mod=8):
    r'''将[data_min, data_max]中的数按照是否被mod整除分成两个列表'''
    train_lst, test_lst = [], []
    for i in range(data_min, data_max):
        if i % mod == 0:
            test_lst.append(i)
        else: 
            train_lst.append(i)

    return train_lst, test_lst

def generate_sequence(args, dataset, mode=1):
    r'''生成单个句子'''

    # 首先生成长度为句长的随机数列表作为句子
    seq = generate_random_list(args.seq_len+1, args.data_min, args.data_max)
    
    # 根据具体任务修改句子中相应的元素

    # 上下文任务
    if args.target == "context":
        seq = context_seq(args, seq, dataset, mode=mode)

    # 上下文任务
    elif args.target == "context2":
        seq = context_seq2(args, seq, dataset, mode=mode)
    
    # 复合函数任务
    elif args.target == "composition":
        seq = composition_seq(args, seq, dataset)
    
    # 多任务同时训练
    elif args.target == 'multitask':
        seq = multitask_seq(args, seq, dataset)
    
    # 简单任务
    elif args.target == '3x_to_x':
        seq = task_3x_to_x_seq(args, seq, dataset)

    # 近义词任务
    elif args.target == 'near_synonym':
        # seq = near_synonym_seq(args, dataset, seq, mode=mode)
        seq = near_synonym_seq_specific(args, dataset, seq, mode=mode)
    
    return seq


def get_data(args, return_dict=False):
    # args 是一个字典，其中包含了许多参数
    r'''
    Required:
        args: {'data_min', 'data_max', 'seq_len', 'batch_size', 
                'train_data_size', 'test_data_size', 'target', 
                'data_mode', 'data_percent', 'data_name' 'data_mask'}
        train/test_seq_group: 以字典形式保存了所有训练/测试集指定类型的句子列表
        train/test_seq_list: 若某些数据类型mask=1，则不会加入到train/test_seq_list中
        train/test_data_loader: 用train/test_seq_list转化来的训练/测试集的DataLoader
    '''
    # 训练集和测试集中组成句子的单词列表？？？？？？？
    variable_train_lst, variable_test_lst = generate_mod_list(args.data_min, args.data_max, args.seq_len-1)

    # 首先将args.data_percent归一化
    # data_percent参数用于决定生成的不同类型数据在测试集和训练集中的数量
    percent_list = np.array(args.data_percent)
    percent_list = percent_list / np.sum(percent_list)
    percent_list = percent_list.tolist()

    # data_mode是干嘛的？？？
    # data_mask:一个列表，表示是否屏蔽对应类型数据的生成。如果为 0，则不生成对应类型的数据
    # 检查args.data_percent, args.data_mode和args.data_name, args.data_mask的长度是否一致
    if len(args.data_percent) != len(args.data_mode) or len(args.data_percent) != len(args.data_name) or len(args.data_percent) != len(args.data_mask):
        raise Exception('args.data_percent, args.data_mode和args.data_name, args.data_mask的长度不一致')


    # 测试集
    test_seq_list = []
    test_seq_group = {}
    for percent, mode, name, mask in zip(percent_list, args.data_mode, args.data_name, args.data_mask):
        tmp_test_seq_list = [generate_sequence(args, variable_test_lst, mode=mode) for _ in range(math.ceil(args.test_data_size * percent))]

        test_seq_group[name] = list(tmp_test_seq_list)
        # 如果mask=0，就将这些句子加入到测试集中
        if mask == 0:
            test_seq_list = test_seq_list + tmp_test_seq_list

    # 训练集
    train_seq_list = []
    train_seq_group = {}
    for percent, mode, name, mask in zip(percent_list, args.data_mode, args.data_name, args.data_mask):
        tmp_train_seq_list = [generate_sequence(args, variable_train_lst, mode=mode) for _ in range(math.ceil(args.train_data_size * percent))]

        train_seq_group[name] = tmp_train_seq_list
    
        if mask == 0:
            train_seq_list = train_seq_list + tmp_train_seq_list


    # 将列表转换为numpy数组
    test_seq_list, train_seq_list = np.array(test_seq_list), np.array(train_seq_list)

    # 将数据集转换为DataLoader
    train_dataset = MyDataSet(train_seq_list)
    train_data_loader = Data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, 
                                        drop_last=True, collate_fn=train_dataset.padding_batch)

    test_dataset = MyDataSet(test_seq_list)
    test_data_loader = Data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, 
                                       drop_last=True, collate_fn=test_dataset.padding_batch)
    # collate_fn=test_dataset.padding_batch？？？？
    

    if return_dict:
        datas = {'train_data_loader': train_data_loader, 'test_data_loader': test_data_loader, 
                'train_seq_group': train_seq_group, 'test_seq_group': test_seq_group, 
                'train_seq_list': train_seq_list, 'test_seq_list': test_seq_list}
        return datas
    else:
        return train_data_loader, test_data_loader, train_seq_group, test_seq_group, train_seq_list, test_seq_list



