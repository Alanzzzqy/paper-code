import re
import matplotlib.pyplot as plt


with open('gradient_info.txt', 'r') as file:
    data_str = file.read().splitlines()

pattern = r'Epoch: (\d+), Batch: 1, Gradients:'
matched_indices = []

# 遍历每一行
for index, line in enumerate(data_str):
    # 使用正则表达式匹配当前行
    if re.match(pattern, line):
        matched_indices.append(index)

layer0_index = [x+5 for x in matched_indices]
layer1_index = [x+15 for x in matched_indices]
layer2_index = [x+25 for x in matched_indices]
layer3_index = [x+35 for x in matched_indices]

layer0 = []
for i in layer0_index:
    # print(data_str[i])
    # print(float(data_str[i][72:]))
    layer0.append(float(data_str[i][72:]))

layer1 = []
for i in layer1_index:
    # print(data_str[i])
    # print(float(data_str[i][72:]))
    layer1.append(float(data_str[i][72:]))

layer2 = []
for i in layer2_index:
    # print(data_str[i])
    # print(float(data_str[i][72:]))
    layer2.append(float(data_str[i][72:]))

layer3 = []
for i in layer3_index:
    # print(data_str[i])
    # print(float(data_str[i][72:]))
    layer3.append(float(data_str[i][72:]))

x = list(range(0, 981, 20))

# 设置图表标题和标签
plt.figure(figsize=(6, 4))
plt.plot(x, layer0, label='layer0')
plt.plot(x, layer1, label='layer1')
plt.plot(x, layer2, label='layer2')
plt.plot(x, layer3, label='layer3')

plt.title('W_V Gradients L2 Norm for Each Layer')
plt.xlabel('Epochs')
plt.ylabel('Gradient L2 Norm')
plt.yscale("log")
plt.legend()
plt.grid()

# plt.show()
plt.savefig('W_V Gradient.png', dpi=300)
