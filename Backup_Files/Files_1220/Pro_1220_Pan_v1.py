import matplotlib.pyplot as plt
import numpy as np


def read_file(file_path, delimiter=" "):
    array_2d = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 移除行尾的换行符并按分隔符分割数据
                row = line.strip().split(delimiter)
                # 将数据转换成适当的类型（如整数或浮点数）
                array_2d.append([float(value) if '.' in value else int(value) for value in row])
    except FileNotFoundError:
        print(f"Error: file of {file_path} cannot been found")
    except ValueError as e:
        print(f"error of data-type: {e}")
    return array_2d

# Prepare data
data = np.array(read_file('./ori_data/f_value.txt', " "))
# data = read_file('./ori_data/f_value.txt', " ")

# for row in data:
#     print(row)

# X-axis
generations = np.arange(1, len(data) + 1)
colors = ['blue', 'orange', 'green', 'red']
labels = ['Max Value', 'Min Value', 'Average Value', 'Deviation']

# Plot the data
plt.figure(figsize=(12, 6))
for i in range(4):
    plt.plot(generations, data[:, i], label=labels[i], color=colors[i])

# Add labels and legend
plt.title('Performance of f1_value Over Generations')
plt.xlabel('Generation')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
