import os
import numpy as np
# from data import configure as c
import configure as c
import librosa
import pywt
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 得到wav文件列表
def file_name(path, postfix='.txt'):
    F = []
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            if os.path.splitext(file)[1] == postfix:
                F.append(file)  # 将所有的文件名添加到F列表中
    return F  # 返回F列表

    # 得到除去拓展名之后的文件的名字


def get_file_name(path):
    B = []
    for root, dirs, files in os.walk(path):
        for file in files:
            B.append(os.path.splitext(file)[0])  # 将文件名和拓展名分开
    return B  # 返回B列表


def calwavelets(y):  # 计算小波分解
    print('采样点的长度', len(y))
    N = int((len(y) - 2048) / 441 + 1)  # 帧数
    T = 0
    i = 0
    num = 0  # 计数2048
    space = []
    per_frame = []  # 存放每一帧数据  # 分帧
    while i < len(y) - 1:
        num += 1  # 计数，记2048
        space.append(y[i])
        i += 1
        if (num % 2048) == 0:  # 2048个采样点为一帧
            T += 1
            per_frame.append(space)
            i = 2047 * T - 1606 * T
            # plt.plot(space)
            # plt.show()
            space = []
            num = 0
        elif i == len(y) - 1 and (num % 2048) != 0:
            break
    print('帧长N:', N, 'per_frame:', len(per_frame))

    # print('energy:\n', energy)
    feature = []  # 存放每帧语音信号构成的特征矢量 Ｙ（ｎ）＝ ［Ｅ１，Ｅ２，Ｅ３，Ｅ４，Ｅ５，Ｅ６，Ｅｍ，σ２］Ｔ
    per_frame_feature = []
    L = []
    for j in range(len(per_frame)):
        # 加窗
        hanwindow = np.hanning(2048)  # 调用汉明窗，把参数帧长传递进去
        signalwindow = per_frame[j] * hanwindow  # 第一帧乘以汉明窗
        # plt.plot(per_frame[j])
        # plt.plot(signalwindow)
        # plt.show()
        A5, D1, D2, D3, D4, D5 = pywt.wavedec(signalwindow, 'db4', mode='symmetric', level=5, axis=-1)
        E_A5 = np.mean(np.abs(A5))
        L.append(E_A5)
        E_D1 = np.mean(np.abs(D1))
        L.append(E_D1)
        E_D2 = np.mean(np.abs(D2))
        L.append(E_D2)
        E_D3 = np.mean(np.abs(D3))
        L.append(E_D3)
        E_D4 = np.mean(np.abs(D4))
        L.append(E_D4)
        E_D5 = np.mean(np.abs(D5))
        L.append(E_D5)

        Em = np.mean(L)  # 均值
        Var = np.var(L)  # 方差

        per_frame_feature.append(E_A5)
        per_frame_feature.append(E_D1)
        per_frame_feature.append(E_D2)
        per_frame_feature.append(E_D3)
        per_frame_feature.append(E_D4)
        per_frame_feature.append(E_D5)
        per_frame_feature.append(Em)
        per_frame_feature.append(Var)

        feature.append(per_frame_feature)
        per_frame_feature = []
        L = []
    print('feature:', len(feature))
    return feature


class AUDIOPROCESS:
    def __init__(self, wav_list, file_list):
        self.Y = []  # 存放不同音频帧数
        self.SR = []  # 存放不同音频采样率
        self.wav = wav_list  # 盛放wav文件的列表
        self.filename = file_list  # 盛放除去拓展名之后的文件的名字，目的是以源文件名保存

    # 获取所有音频文件的时间长度和采样率,及特征
    def WAVFEATURE(self, wav_path, save_path):
        self.wav_path = wav_path  # 原始wav文件的路径
        self.save_path = save_path  # 提取特征之后的存储路径
        for j in range(len(self.wav)):
            y, sr = librosa.load(os.path.join(wav_path, self.wav[j]), sr=None)
            y = librosa.resample(y, sr, 44100)
            # a = len(y)
            self.SR.append(sr)
            # 分帧
            N=int((len(y) - 2048) /441 + 1)   # 帧数
            self.Y.append(N)  # 存放帧数
            # 提取小波特征
            WA = calwavelets(y)
            WA = np.array(WA)
            print("第{}个文件的feature:".format(j), WA.shape, self.wav[j])
            print('************************************//////////////////////////***************************')
            # scipy.io.savemat(save_path + self.filename[j] + '.mat', {'feature': feature})
            np.savetxt(save_path + self.filename[j] + '.csv', WA, fmt='%10.5f', delimiter=',')
            j += 1

    # 提取语音标签
    def AUDIOLABEL(self, txt_path, txtfile, save_path):
        self.txtfile = txtfile  # 盛放txt文件

        # 将txtfile文件以列表形式读取到file_test中
        self.file_test = []
        for f in self.txtfile:
            file_path = os.path.join(txt_path, f)
            txt_contest = np.loadtxt(file_path)
            self.file_test.append(txt_contest)
        print("file_text", type(self.file_test))
        for n in range(len(self.Y)):  # len(self.Y) 一共存放了几个文件
            t = self.Y[n]  # 取出第n个音频的帧数
            print("第{}个文件长度".format(n + 1), t, txtfile[n])
            # 将a[]中先存放t帧数的0
            a = []
            i = 0

            while i <= t - 1:
                a.append(0)
                i += 1
            # ---------------平铺列表-------------------------------------
            b = self.file_test[n].flatten()
            # print("列表长度:", len(b))
            # -----------------------------------------------------------
            space = []  # 定义一个临时空间
            j = 0
            while j <= len(b) - 1:  # 定义一个j，最大不超过列表b的长度
                space.append(b[j])
                space.append(b[j + 1])  # 取出列表里的第一对值放到临时空间space
                # print(space)
                # k1 = math.floor(space[0] / (2048 / 44100 - 0.01))  # 定义k为space中的第一个值
                # k2 = math.floor(space[1] / (2048 / 44100 - 0.01))
                k1 = math.floor((44100 * space[0] - 1607) / 441)  # 定义k为space中的第一个值
                k2 = math.floor((44100 * space[1] - 1607) / 441)
                # print("第{}个值".format(j), len(a))
                while k1 <= k2:
                    if int(k1) > len(a) -1:
                        print("这是第{}个文件".format(n + 1), int(k1), int(k2), len(a),
                              "IndexError: list assignment index out of range!")
                        exit()
                    else:
                        a[int(k1)-1] = 1  # 更新列表对应位置a的值
                        k1 += 1
                space.clear()  # 清空空间
                j += 2  # 取列表的第二对值
            m = np.array(a).transpose()
            # print(m)
            # --------（reshape(-1,1)按列）保存处理过的文件----------------------
            # scipy.io.savemat(save_path + self.filename[n] + '.mat', {'label': m.reshape(-1, 1)})
            np.savetxt(save_path + self.filename[n] + '.csv', m, fmt='%10.5f', delimiter=',')
            n += 1
