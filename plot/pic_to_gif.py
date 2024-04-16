import os
import imageio.v2 as imageio

# 设置图片文件夹和输出动图文件名

# 获取当前脚本所在的文件夹路径
current_folder = os.path.dirname(__file__)
# 获取所有文件夹名称
subdirectories = [d for d in os.listdir(current_folder) if os.path.isdir(os.path.join(current_folder, d))]
# 提取符合条件的文件夹名称
desired_folders = [folder for folder in subdirectories if folder.count('-') == 4]
# 输出符合条件的文件夹名称
print(desired_folders)

if not os.path.exists('attn_gif'):
    os.makedirs('attn_gif')

for i in desired_folders:
    f_name = i
    image_folder = f'{f_name}/attn_pic'
    save_gif_path = f'attn_gif/{f_name}.gif'
    # 获取文件夹中的所有图片文件
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.png')]

    # 按照文件名进行排序
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    # 读取图片并将它们添加到动图
    with imageio.get_writer(save_gif_path, duration=10.0, fps=1) as writer:
        for image in images:
            img = imageio.imread(image)
            writer.append_data(img)

    print(f'动图已保存至 {save_gif_path}')
