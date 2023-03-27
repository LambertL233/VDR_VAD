import os
import configure as c
import numpy as np


class DATA:
    def __init__(self, label_file_list, feature_file_list):
        self.label_file_list = label_file_list  # 需要处理的标签文件
        self.feature_file_list = feature_file_list  # 需要处理的特征文件

    def CONCATENATE(self, lab_path, fea_path, data_save, name1, name2):
        self.lab_path = lab_path  # 存放单个新标签的路径
        self.fea_path = fea_path  # 存放单个新特征的路径
        self.data_save = data_save  # 整合的单一文件存放路径
        self.name1 = name1  # 标签存储的文件名
        # self.name1_1 = name1_1  # 标签字典的关键字
        self.name2 = name2  # 特征存储的文件名
        # self.name2_2 = name2_2  # 特征字典关键字

        label = []
        for l in self.label_file_list:
            file_path = os.path.join(lab_path, l)
            y = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=0)  # CSV文件转化为数组
            # y = loadmat(file_path)
            # y = y['label']
            label.append(y)
        label = np.concatenate(label)
        # scio.savemat(data_save + self.name1 + '.mat', {self.name1_1: label})
        np.savetxt(data_save + self.name1 + '.csv', label, fmt='%10.5f', delimiter=',')
        print('label shape:', label.shape)
        feature = []
        for f in self.feature_file_list:
            file_path = os.path.join(fea_path, f)
            y = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=0)  # CSV文件转化为数组
            # y = loadmat(file_path)
            # y = y['new_feature']
            # print(y.shape)
            print(y.shape)
            feature.append(y)
        feature = np.concatenate(feature, axis=0)
        # scio.savemat(data_save + self.name2 + '.mat', {self.name2_2: feature})
        np.savetxt(data_save + self.name2 + '.csv', feature, fmt='%10.5f', delimiter=',')
        print('feature shape:', feature.shape)


def read_data(name_f, name_l, save_path, name):
    F = np.loadtxt(open(c.datafile + name_f, "rb"), delimiter=",", skiprows=0)  # CSV文件转化为数组
    # F = scipy.io.loadmat(os.path.join(c.datafile, name_f))
    # F = F[key_f]
    print("特征维度：", F.shape)
    L = np.loadtxt(open(c.datafile + name_l, "rb"), delimiter=",", skiprows=0)  # CSV文件转化为数组
    L = np.expand_dims(L, axis=1)
    # L = scipy.io.loadmat(os.path.join(c.datafile, name_l))
    # L = L[key_l]
    print("标签维度：", L.shape)
    mix = np.concatenate((F, L), axis=1)
    print("合并后的维度:", mix.shape)
    np.savetxt(save_path + name + '.csv', mix, fmt='%10.5f', delimiter=',')
    # scipy.io.savemat(save_path + name + '.mat', {'data': mix})
