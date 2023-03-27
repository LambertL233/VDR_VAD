import numpy as np


class DataAlignment:
    def __init__(self, FEA_FILE, LAB_FILE):
        self.FEA_FILE = FEA_FILE  # 需要处理的特征文件
        self.LAB_FILE = LAB_FILE  # 需要处理的标签文件

    def dataprocess(self, FEA_PATH, LAB_PATH, FEA_SAVE_PATH, LAB_SAVE_PATH):
        self.FEA_PATH = FEA_PATH  # 原始feature文件路径
        self.LAB_PATH = LAB_PATH  # 原始label文件加载路径
        self.FEA_SAVE_PATH = FEA_SAVE_PATH  # 处理后的feature的mat文件存储路径
        self.LAB_SAVE_PATH = LAB_SAVE_PATH  # 处理后的label的mat文件存储路径
        for i in range(len(self.FEA_FILE)):
            F_Array = np.loadtxt(open(self.FEA_PATH + self.FEA_FILE[i], "rb"), delimiter=",", skiprows=0)  # CSV文件转化为数组
            X_F_Shape = F_Array.shape[0]
            # context = loadmat(self.FEA_PATH + self.FEA_FILE[i])  # loadmat 以字典形式读取文件内容
            # print(context.keys())
            # F_Array = context["feature"]  # 根据（key）关键字得到存放的数据文件数组
            # F_shape = F_Array[:, :].shape  # 字典的到的是包括列表维度的三维数组，这里的目的是去掉列表的一维度，要数据的二维度
            # X_F_Shape = F_shape[0]

            # lcontext = loadmat(self.LAB_PATH + self.LAB_FILE[i])
            # L_Array = lcontext["label"]
            # L_shape = L_Array[:].shape
            # X_L_Shape = L_shape[0]
            L_Array = np.loadtxt(open(self.LAB_PATH + self.LAB_FILE[i], "rb"), delimiter=",", skiprows=0)  # CSV文件转化为数组
            X_L_Shape = L_Array.shape[0]

            if X_F_Shape > X_L_Shape:
                subtractor = X_F_Shape - X_L_Shape  # 取出特征\标签相差的长度值
                # x1 = F_Array.tolist()  # 数组先转化为列表，便于进行切片操作
                y1 = int(subtractor / 2)  # 将相差的长度值一分为2，便于前面数组删一部分，后面部分删一半
                # x2 = x1[::(y1 - 1)]  # 取出前半部分需要删除的数据
                y2 = subtractor - y1
                A = np.delete(F_Array, np.s_[:y1], axis=0)
                B = np.delete(A, np.s_[-y2:], axis=0)
                # x1[::(y1 - 1)][-y2::]
                # y = np.array(x1)
                # print(B)
                print("第{}个修改过的文件的特征维度:".format(i), B.shape)
                print("第{}个文件的标签维度:".format(i), L_Array.shape)
                # scio.savemat(FEA_SAVE_PATH + self.FEA_FILE[i] + '.mat', {'new_feature': B})
                np.savetxt(FEA_SAVE_PATH + self.FEA_FILE[i] + '.csv', B, fmt='%10.5f', delimiter=',')

            elif X_F_Shape < X_L_Shape:
                subtractor = X_L_Shape - X_F_Shape  # 取出特征\标签相差的长度值
                # x1 = list(L_Array)  # 数组先转化为列表，便于进行切片操作
                y1 = int(subtractor / 2)  # 将相差的长度值一分为2，便于前面数组删一部分，后面部分删一半
                y2 = subtractor - y1
                # x2 = x1[::y1 - 1]  # 取出前半部分需要删除的数据
                # del x1[::(y1 - 1)]  # 取出前半部分需要删除的数据
                # y2 = subtractor - y1
                # # x3 = x1[-y2::]  # 取出后半部分需要删除出的数据
                # del x1[-y2::]  # 取出后半部分需要删除出的数据(还是有问题，这样也没删掉………………)
                # y = np.array(x1)
                A = np.delete(F_Array, np.s_[:y1], axis=0)  # 删除前半部分数据
                B = np.delete(A, np.s_[-y2:], axis=0)  # 删除后半部分数据
                # scio.savemat(LAB_SAVE_PATH + self.LAB_FILE[i] + '.mat', {'new_label': B})
                np.savetxt(LAB_SAVE_PATH + self.LAB_FILE[i] + '.csv', B, fmt='%10.5f', delimiter=',')
            else:
                pass

            i += 1


'''
    注意：x2 = x1[::y1 - 1]  # 取出前半部分需要删除的数据
         del x2
         与
         del x1[::y1 - 1]
         是不一样的，前者的输出x1还是原来的输出x1,后者的输出x1才是删除一部分数据之后的x1
'''
