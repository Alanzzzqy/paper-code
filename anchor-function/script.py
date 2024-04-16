import os

# # seed_list = range(1, 4)
# seed_list = [1]
# # lr = 2e-5
# lr = 1e-4
# # N_train_list = [100, 200, 300, 400, 500, 600, 700, 800]
# N_train_list = [800]
# # N_train_list = [300, 400, 500, 600, 700]
# # N_train_list = [2000, 1500, 900, 600]
# # N_train_list = [300, 500, 1000, 2000]
# # N_train_list = [300]
# # N_train_list = [50000]
# model='GPT'
# # model = 'DNN'
# # target = 'composition'
# # target = 'context'
# # target = 'multitask'
# target = '3x_to_x'

# scheduler = 'none'
# # scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'

# batch_size = 100

# for seed in seed_list:
#     for N_train in N_train_list:
#         # os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model}')
#         # os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#         #           -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs 100')
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size}')

# # GPT_multtask_10个任务一起做
# seed_list = [1]
# lr = 2e-5
# N_train_list = [500, 1000, 2000, 5000, 8000]
# batch_size = 100
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# target = 'multitask'
# dir_suffix = 'multitask_10个任务一起做'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix}')


# # GPT_1层4head
# seed_list = [1]
# target = '3x_to_x'
# lr = 1e-4
# scheduler = 'none'
# model='GPT'
# batch_size = 100
# N_train_list = [10000]
# dir_suffix = '1层_1head_3x_to_x'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 1 -nh 1 -bs {batch_size} -dir_suffix {dir_suffix}')


# # GPT_复合函数
# seed_list = [1]
# target = 'composition'
# dir_suffix = '复合函数_4层_warmup_cos_lr_2e-5'
# lr = 2e-5
# N_train_list = [500, 1000, 2000, 4000, 7000, 10000]
# batch_size = 100
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix}')
        
# # GPT_复合函数
# seed_list = [1]
# target = 'composition'
# dir_suffix = '复合函数_4层_warmup_cos_lr_1e-4'
# lr = 1e-4
# N_train_list = [500, 1000, 2000, 4000, 7000, 10000]
# batch_size = 100
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix}')


# # DNN 3x_to_x 4层
# seed_list = [1]
# target = '3x_to_x'
# lr = 1e-4
# scheduler = 'none'
# model='DNN'
# batch_size = 10
# N_train_list = [400, 800, 1200, 1600]
# dir_suffix = '4层_3x_to_x'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 4 -nh 1 -bs {batch_size} -dir_suffix {dir_suffix}')
        
# # DNN 3x_to_x 2层
# seed_list = [1]
# target = '3x_to_x'
# lr = 1e-4
# scheduler = 'none'
# model='DNN'
# batch_size = 100
# N_train_list = [400, 800, 1200, 1600]
# dir_suffix = '2层_3x_to_x'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 2 -nh 1 -bs {batch_size} -dir_suffix {dir_suffix}')


# # GPT 上下文学习
# seed_list = [1]
# target = 'context2'
# dir_suffix = '上下文学习_2prompt_4层_warmup_cos_lr_2e-5'
# lr = 2e-5
# N_train_list = [500, 1000, 2000, 4000]
# # N_train_list = [500]
# batch_size = 100
# percent = [0.5, 0.5]
# data_mode = ['abab', 'abba']
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'/bin/python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
#                   -dp {percent[0]} {percent[1]} -dmode {data_mode[0]} {data_mode[1]}')

# ------------------ 后边开始用新的代码 ------------------
# # GPT 近义词学习
# seed_list = [1]
# target = 'near_synonym'
# dir_suffix = '近义词学习11112_4层_warmup_cos_lr_2e-5'
# lr = 2e-5
# # N_train_list = [2000, 4000, 10000, 20000]
# N_train_list = [2000]
# batch_size = 100
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# model = 'GPT'

# dname, dmode, dpercent, dmask, dshow = [], [], [], [], []
# for i in [1,2,3,4]:
#     for j in [1,2,3,4]:
#         prompt = f'{i}{j}'
#         dname.append(prompt)
#         dmode.append(prompt)
#         dpercent.append(4)
#         if prompt in ['34', '94']:
#             dmask.append(1)
#             dshow.append(1)
#         else:
#             dmask.append(0)
#             dshow.append(0)

# for i in [1,2,9,4]:
#     for j in [1,2,9,4]:
#         prompt = f'{i}{j}'
#         dname.append(prompt)
#         dmode.append(prompt)
#         dpercent.append(1)
#         if prompt in ['34', '94']:
#             dmask.append(1)
#             dshow.append(1)
#         else:
#             dmask.append(0)
#             dshow.append(0)
        
# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dmask = ' '.join(map(str, dmask))
# dshow = ' '.join(map(str, dshow))

# for N_train in N_train_list:
#     os.system(f'/bin/python -m main -N_train {N_train} -seed 1 -func {target} -lr {lr} -m {model}\
#                 -scheduler {scheduler} -ne 4000 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} \
#                 -dmode {dmode} -dp {dp} -dn {dn} -dmask {dmask} -dshow {dshow}')


# # GPT 3x_to_x 4层
# # seed_list = [6,7,8,9,10]
# seed_list = [1]
# target = '3x_to_x'
# lr = 2e-5
# model='GPT'
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# batch_size = 50
# # N_train_list = [550, 600, 650, 700, 750, 800]
# N_train_list = [200]
# dir_suffix = '3x_to_x_for_paper'
# device='0'
# print("test")
# for seed in seed_list:
#     for N_train in N_train_list:
#         os.system(f'python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
#                   -scheduler {scheduler} -ne 500 -nl 4 -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} -device {device}')

# GPT 3x_to_x 4层
target = '3x_to_x'
lr = 2e-5
scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
batch_size = 50
device='0'
dir_suffix = '3x_to_x_for_paper'
n_epoch = 1000
save_model_epoch = 1000 


exp_name = "mask_x3_exp" #实验名称
exp_index = "3" #实验组号
seed_list = [6,7,8,9,10]
N_train_list = [2000]
n_layers = 2
yes_or_no = "yes" #是否含有某些模块  
n_heads = 4


if yes_or_no=="yes": model='GPT'
elif yes_or_no=="no": model='GPT_without'

print("test")
for seed in seed_list:
    for N_train in N_train_list:
        os.system(f'python -m main -N_train {N_train} -seed {seed} -m {model} -func {target} -lr {lr} \
                  -scheduler {scheduler} -ne {n_epoch} -nl {n_layers} -nh {n_heads} -bs {batch_size} -dir_suffix {dir_suffix} -device {device}\
                  --exp_name {exp_name} --exp_index {exp_index} --yes_or_no {yes_or_no} --save_model_epoch {save_model_epoch}')


