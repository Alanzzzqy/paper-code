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
# if not os.path.exists('pic/uncompare/std'):
#     os.makedirs('pic/uncompare/std')
# if not os.path.exists('pic/uncompare/without'):
#     os.makedirs('pic/uncompare/without')

# 实验参数
exp_index = '4'
N_seed = 10
N_train = 1000
n_layers = 2
n_heads = 1
exp_name = 'head'

# 模型实验组路径
head1_path = '/content/drive/MyDrive/anchor-function/anchor-function/result/GPT_3x_to_x_for_paper/forward_exp/4'
head2_path = '/content/drive/MyDrive/anchor-function/anchor-function/result/GPT_3x_to_x_for_paper/norm_exp/2'
head3_path = ''
head4_path = '/content/drive/MyDrive/anchor-function/anchor-function/result/GPT_3x_to_x_for_paper/norm_exp/1'

# Compare test
for i in range(N_seed):
    file_path = str(i+1)+'-'+str(N_train)+'-'+str(n_layers)+"-yes-"
    pic_title = "seed"+str(i+1)+'-N'+str(N_train)+'-nl'+str(n_layers)
    print(pic_title)


    fig2, axs2=plt.subplots(1,2,figsize=(12,4))
    head1_loss = np.load(os.path.join(head1_path, file_path+'1', 'loss', 'test_loss_his.npy'))
    head2_loss = np.load(os.path.join(head2_path, file_path+'2', 'loss', 'test_loss_his.npy'))
    head3_loss = np.load(os.path.join(head3_path, file_path+'3', 'loss', 'test_loss_his.npy'))
    head4_loss = np.load(os.path.join(head4_path, file_path+'4', 'loss', 'test_loss_his.npy'))

    head1_acc = np.load(os.path.join(head1_path, file_path+'1', 'loss', 'test_acc_his.npy'))
    head2_acc = np.load(os.path.join(head2_path, file_path+'2', 'loss', 'test_acc_his.npy'))
    head3_acc = np.load(os.path.join(head3_path, file_path+'3', 'loss', 'test_acc_his.npy'))
    head4_acc = np.load(os.path.join(head4_path, file_path+'4', 'loss', 'test_acc_his.npy'))

    x=np.arange(1,1001)
    axs2[0].set(xlabel='epoch', ylabel='test_loss', title=pic_title)
    axs2[0].plot(x,head1_loss,x,head2_loss,x,head3_loss,x,head4_loss) # 多线一起画，自动颜色
    axs2[0].legend(['1head','2heads','3heads','4heads'])
    axs2[0].grid()
    axs2[0].set_yscale('log')

    x=np.arange(0,11)*100
    axs2[1].set(xlabel='epoch', ylabel='test_acc', 	title=pic_title)
    axs2[1].plot(x,head1_acc,'o-',x,head2_acc,'o-',x,head3_acc,'o-',x,head4_acc,'o-') # 多线一起画，自动颜色
    axs2[1].legend(['1head','2heads','3heads','4heads'])
    axs2[1].grid()

    plt.tight_layout()
    # plt.savefig(pic_title,dpi=300)
    plt.savefig(os.path.join('pic/compare/test', pic_title + '.png'), dpi=300)
    plt.close()
    # plt.show()

# Compare train
for i in range(N_seed):
    file_path = str(i+1)+'-'+str(N_train)+'-'+str(n_layers)+"-yes-"
    pic_title = "seed"+str(i+1)+'-N'+str(N_train)+'-nl'+str(n_layers)
    print(pic_title)


    fig2, axs2=plt.subplots(1,2,figsize=(12,4))
    head1_loss = np.load(os.path.join(head1_path, file_path+'1', 'loss', 'train_loss_his.npy'))
    head2_loss = np.load(os.path.join(head2_path, file_path+'2', 'loss', 'train_loss_his.npy'))
    head3_loss = np.load(os.path.join(head3_path, file_path+'3', 'loss', 'train_loss_his.npy'))
    head4_loss = np.load(os.path.join(head4_path, file_path+'4', 'loss', 'train_loss_his.npy'))

    head1_acc = np.load(os.path.join(head1_path, file_path+'1', 'loss', 'train_acc_his.npy'))
    head2_acc = np.load(os.path.join(head2_path, file_path+'2', 'loss', 'train_acc_his.npy'))
    head3_acc = np.load(os.path.join(head3_path, file_path+'3', 'loss', 'train_acc_his.npy'))
    head4_acc = np.load(os.path.join(head4_path, file_path+'4', 'loss', 'train_acc_his.npy'))

    x=np.arange(1,1001)
    axs2[0].set(xlabel='epoch', ylabel='train_loss', title=pic_title)
    axs2[0].plot(x,head1_loss,x,head2_loss,x,head3_loss,x,head4_loss) # 多线一起画，自动颜色
    axs2[0].legend(['1head','2heads','3heads','4heads'])
    axs2[0].grid()
    axs2[0].set_yscale('log')

    x=np.arange(0,11)*100
    axs2[1].set(xlabel='epoch', ylabel='train_acc', 	title=pic_title)
    axs2[1].plot(x,head1_acc,'o-',x,head2_acc,'o-',x,head3_acc,'o-',x,head4_acc,'o-') # 多线一起画，自动颜色
    axs2[1].legend(['1head','2heads','3heads','4heads'])
    axs2[1].grid()

    plt.tight_layout()
    # plt.savefig(pic_title,dpi=300)
    plt.savefig(os.path.join('pic/compare/train', pic_title + '.png'), dpi=300)
    plt.close()
    # plt.show()