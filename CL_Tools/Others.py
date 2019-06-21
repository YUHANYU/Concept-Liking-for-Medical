#-*-coding:utf-8-*-

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import os


# 把输入数据集分为训练，验证和测试三个部分
def train_val_test_1(data, train_size=0.9, test_size=0.8, shuffle=True):
    train_data, val_test_data = train_test_split(
        data, train_size=train_size, test_size=1 - train_size, shuffle=True, random_state=2019)
    val_data, test_data = train_test_split(
        val_test_data, train_size=1 - test_size, test_size=test_size, shuffle=True, random_state=2019)

    return train_data, val_data, test_data


# 把输入数据集和目标数据集分为训练，验证和测试三个部分
def train_val_test_2(data_1, data_2, train_size=0.9, test_size=0.8, shuffle=True):
    train_data_x, val_test_data_x, train_data_y, val_test_data_y = train_test_split(
        data_1, data_2, train_size=train_size, test_size=1 - train_size, shuffle=shuffle, random_state=2019
    )
    val_data_x, test_data_x, val_data_y, test_data_y = train_test_split(
        val_test_data_x, val_test_data_y, train_size=1 - test_size, test_size=test_size, shuffle=True,
        random_state=2019
    )

    return train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y


# 可视化损失
def show_loss(x, y, picture_name, color='g'):
    lines = plt.plot(x, y, '.-')
    plt.setp(lines, color=color, linewidth=.5)
    plt.title('Train loss in every steps of epochs')
    plt.ylabel('loss')
    plt.xlabel('each steps of every epochs')

    # plt.show()
    # 当前项目路径
    project_path = os.getcwd()
    plt.savefig((project_path + '/CL_Save_Pictures/' + picture_name + '.png'), dpi=100)
    print(project_path + '/CL_Save_Pictures/' + picture_name + '.png')
