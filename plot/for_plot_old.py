import numpy as np
import matplotlib.pyplot as plt
import os

# 字体格式设置
plt.rcParams['font.sans-serif'] = ['Arial']
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的文件夹
if not os.path.exists('pic/compare/test'):
    os.makedirs('pic/compare/test')
if not os.path.exists('pic/compare/train'):
    os.makedirs('pic/compare/train')
if not os.path.exists('pic/uncompare/std'):
    os.makedirs('pic/uncompare/std')
if not os.path.exists('pic/uncompare/without'):
    os.makedirs('pic/uncompare/without')

# 实验参数
exp_index = '4'
N_seed = 10
N_train = 1000
n_layers = 2
n_heads = 1
exp_name = 'forward'

# 标准模型的绝对路径
std_file_absolute_path = '/content/drive/MyDrive/anchor-function/anchor-function/result/GPT_3x_to_x_for_paper/norm_exp'


# Compare test
for i in range(N_seed):
    std_file = str(i+1)+'-'+str(N_train)+'-'+str(n_layers)+"-yes-"+str(n_heads)
    exp_file = str(i+1)+'-'+str(N_train)+'-'+str(n_layers)+"-no-"+str(n_heads)
    pic_title = "seed"+str(i+1)+'-N'+str(N_train)+'-nl'+str(n_layers)+'-nh'+str(n_heads)
    # print(std_file)
    # print(exp_file)
    print(pic_title)


    fig2, axs2=plt.subplots(1,2,figsize=(12,4))
    loss_std=np.load(std_file+'/loss/test_loss_his.npy')
    # loss_std=np.load(os.path.join(std_file_absolute_path, exp_index, std_file,'loss', 'test_loss_his.npy'))
    loss_without_norm=np.load(exp_file+'/loss/test_loss_his.npy')
    acc_std=np.load(std_file+'/loss/test_acc_his.npy')
    # acc_std=np.load(os.path.join(std_file_absolute_path, exp_index, std_file, 'loss', 'test_acc_his.npy'))
    acc_without_norm=np.load(exp_file+'/loss/test_acc_his.npy')

    x=np.arange(1,1001)
    axs2[0].set(xlabel='epoch', ylabel='test_loss', title=pic_title)
    axs2[0].plot(x,loss_std,x,loss_without_norm) # 多线一起画，自动颜色
    axs2[0].legend(["std",'without_'+exp_name])
    axs2[0].grid()
    axs2[0].set_yscale('log')

    x=np.arange(0,11)*100
    axs2[1].set(xlabel='epoch', ylabel='test_acc', 	title=pic_title)
    axs2[1].plot(x,acc_std,'o-',x,acc_without_norm,'o-') # 多线一起画，自动颜色
    axs2[1].legend(["std",'without_'+exp_name])
    axs2[1].grid()

    plt.tight_layout()
    # plt.savefig(pic_title,dpi=300)
    plt.savefig(os.path.join('pic/compare/test', pic_title + '.png'), dpi=300)
    plt.close()
    # plt.show()

# Compare train
for i in range(N_seed):
    std_file = str(i+1)+'-'+str(N_train)+'-'+str(n_layers)+"-yes-"+str(n_heads)
    exp_file = str(i+1)+'-'+str(N_train)+'-'+str(n_layers)+"-no-"+str(n_heads)
    pic_title = "seed"+str(i+1)+'-N'+str(N_train)+'-nl'+str(n_layers)+'-nh'+str(n_heads)
    # print(std_file)
    # print(exp_file)
    print(pic_title)


    fig2, axs2=plt.subplots(1,2,figsize=(12,4))
    loss_std=np.load(std_file+'/loss/train_loss_his.npy')
    # loss_std=np.load(os.path.join(std_file_absolute_path, exp_index, std_file, 'loss', 'train_loss_his.npy'))
    loss_without_norm=np.load(exp_file+'/loss/train_loss_his.npy')
    acc_std=np.load(std_file+'/loss/train_acc_his.npy')
    # acc_std=np.load(os.path.join(std_file_absolute_path, exp_index, std_file, 'loss', 'train_acc_his.npy'))
    acc_without_norm=np.load(exp_file+'/loss/train_acc_his.npy')

    x=np.arange(1,1001)
    axs2[0].set(xlabel='epoch', ylabel='train_loss', title=pic_title)
    axs2[0].plot(x,loss_std,x,loss_without_norm) # 多线一起画，自动颜色
    axs2[0].legend(["std",'without_'+exp_name])
    axs2[0].grid()
    axs2[0].set_yscale('log')

    x=np.arange(0,11)*100
    axs2[1].set(xlabel='epoch', ylabel='train_acc', 	title=pic_title)
    axs2[1].plot(x,acc_std,'o-',x,acc_without_norm,'o-') # 多线一起画，自动颜色
    axs2[1].legend(["std",'without_'+exp_name])
    axs2[1].grid()

    plt.tight_layout()
    # plt.savefig(pic_title,dpi=300)
    plt.savefig(os.path.join('pic/compare/train', pic_title + '.png'), dpi=300)
    plt.close()
    # plt.show()

# uncompare exp
for i in range(N_seed):
    std_file = str(i+1)+'-'+str(N_train)+'-'+str(n_layers)+"-yes-"+str(n_heads)
    exp_file = str(i+1)+'-'+str(N_train)+'-'+str(n_layers)+"-no-"+str(n_heads)
    pic_title = "seed"+str(i+1)+'-N'+str(N_train)+'-nl'+str(n_layers)+'-nh'+str(n_heads)
    # print(std_file)
    # print(exp_file)
    print(pic_title)


    fig2, axs2=plt.subplots(1,2,figsize=(12,4))
    loss_test = np.load(exp_file+'/loss/test_loss_his.npy')
    loss_train = np.load(exp_file+'/loss/train_loss_his.npy')
    acc_test = np.load(exp_file+'/loss/test_acc_his.npy')
    acc_train = np.load(exp_file+'/loss/train_acc_his.npy')
   
    x=np.arange(1,1001)
    axs2[0].set(xlabel='epoch', ylabel='loss', title=pic_title)
    axs2[0].plot(x,loss_test,x,loss_train) # 多线一起画，自动颜色
    axs2[0].legend(["test",'train'])
    axs2[0].grid()
    axs2[0].set_yscale('log')

    x=np.arange(0,11)*100
    axs2[1].set(xlabel='epoch', ylabel='acc', 	title=pic_title)
    axs2[1].plot(x,acc_test,'o-',x,acc_train,'o-') # 多线一起画，自动颜色
    axs2[1].legend(["test",'train'])
    axs2[1].grid()

    plt.tight_layout()
    # plt.savefig(pic_title,dpi=300)
    plt.savefig(os.path.join('pic/uncompare/without', pic_title + '.png'), dpi=300)
    plt.close()
    # plt.show()

# uncompare std
for i in range(N_seed):
    std_file = str(i+1)+'-'+str(N_train)+'-'+str(n_layers)+"-yes-"+str(n_heads)
    exp_file = str(i+1)+'-'+str(N_train)+'-'+str(n_layers)+"-no-"+str(n_heads)
    pic_title = "seed"+str(i+1)+'-N'+str(N_train)+'-nl'+str(n_layers)+'-nh'+str(n_heads)
    # print(std_file)
    # print(exp_file)
    print(pic_title)


    fig2, axs2=plt.subplots(1,2,figsize=(12,4))
    loss_test = np.load(std_file+'/loss/test_loss_his.npy')
    loss_train = np.load(std_file+'/loss/train_loss_his.npy')
    acc_test = np.load(std_file+'/loss/test_acc_his.npy')
    acc_train = np.load(std_file+'/loss/train_acc_his.npy')
   
    # loss_test=np.load(os.path.join(std_file_absolute_path, exp_index, std_file,'loss', 'test_loss_his.npy'))
    # loss_train=np.load(os.path.join(std_file_absolute_path, exp_index, std_file, 'loss', 'train_loss_his.npy'))
    # acc_test=np.load(os.path.join(std_file_absolute_path, exp_index, std_file, 'loss', 'test_acc_his.npy'))
    # acc_train=np.load(os.path.join(std_file_absolute_path, exp_index, std_file, 'loss', 'train_acc_his.npy'))
    
    x=np.arange(1,1001)
    axs2[0].set(xlabel='epoch', ylabel='loss', title=pic_title)
    axs2[0].plot(x,loss_test,x,loss_train) # 多线一起画，自动颜色
    axs2[0].legend(["test",'train'])
    axs2[0].grid()
    axs2[0].set_yscale('log')

    x=np.arange(0,11)*100
    axs2[1].set(xlabel='epoch', ylabel='acc', 	title=pic_title)
    axs2[1].plot(x,acc_test,'o-',x,acc_train,'o-') # 多线一起画，自动颜色
    axs2[1].legend(["test",'train'])
    axs2[1].grid()

    plt.tight_layout()
    # plt.savefig(pic_title,dpi=300)
    plt.savefig(os.path.join('pic/uncompare/std', pic_title + '.png'), dpi=300)
    plt.close()
    # plt.show()
