import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy
import os
import math
from spafe.features.gfcc import gfcc
from spafe.features.bfcc import bfcc
from spafe.features.msrcc import msrcc
from mixfeature11 import configuregfcc as F
from mixfeature11 import combine as o


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
        for file in sorted(files):
            B.append(os.path.splitext(file)[0])  # 将文件名和拓展名分开
    return B  # 返回B列表


# 分帧
def enframe(signal, nw, inc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:帧移
    '''
    signal_length = len(signal)  # 信号总长度

    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1

        nf = 1

    else:  # 否则，计算帧的总长度

        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))

    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                           (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    #  win=np.tile(winfunc(nw),(nf,1)) #window窗函数，这里默认取1
    #  return frames*win  #返回帧信号矩阵
    return frames


# librosa.display.waveplot(sig, sr=sr, x_axis='time')  # 原始信号图像
# plt.show()
def compute_features(sig, sr, inc, wlen):
    """
    :param sig: 采样值
    :param sr: 采样率
    :param inc: 帧移
    :param wlen: 窗长/帧移
    :return: mix_feature (帧长 * 特征)
    """
    # 分帧
    N = enframe(sig, wlen, inc)  # 数组（帧数 * 帧长）

    # 计算log常Q变换特征
    # C = np.abs(np.log(librosa.cqt(sig, sr=sr, window=scipy.signal.windows.hann))).transpose()

    # 梅尔倒谱系数
    M = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=20, hop_length=inc).transpose()
    # 一阶差分
    mfcc_delta = librosa.feature.delta(M, width=3, order=1, axis=- 1, mode='interp')
    # 二阶差分
    mfcc_delta2 = librosa.feature.delta(M, order=2)
    # 过零率
    # zcr = librosa.feature.zero_crossing_rate(sig, frame_length=wlen, hop_length=inc, center=True).transpose()

    # bark-frequency cepstral coefﬁcients (BFCC features)
    bfccs = bfcc(sig, fs=sr, num_ceps=20, win_len=wlen / sr, win_hop=inc / sr, win_type='hanning',
                 nfilts=26, nfft=2048, scale='constant', dct_type=2, use_energy=False, lifter=5, normalize=True)
    # 一阶差分
    bfccs_delta = librosa.feature.delta(bfccs, width=3, order=1, axis=- 1, mode='interp')
    # 二阶差分
    bfccs_delta2 = librosa.feature.delta(bfccs, order=2)

    # 从音频信号计算基于Gammatone滤波器听觉模型倒谱特征参数
    gfccs = gfcc(sig=sig, fs=sr, num_ceps=20, pre_emph=0, pre_emph_coeff=0.97, win_len=wlen / sr, win_type="hanning",
                 win_hop=inc / sr, nfilts=26, nfft=2048, low_freq=None, high_freq=None, dct_type=2, use_energy=False,
                 lifter=22, normalize=1)
    # 一阶差分
    gfccs_delta = librosa.feature.delta(gfccs, width=3, order=1, axis=- 1, mode='interp')
    # 二阶差分
    gfccs_delta2 = librosa.feature.delta(gfccs, order=2)
    # 从音频信号计算基于震级的谱根倒谱系数(MSRCC)。
    msrccs = msrcc(sig, fs=sr, num_ceps=20, pre_emph=0, pre_emph_coeff=0.97, win_len=wlen / sr, win_hop=inc / sr,
                   win_type='hanning', nfilts=26, nfft=2048,
                   scale='constant', gamma=-0.14285714285714285, dct_type=2, use_energy=False, lifter=22, normalize=1)
    # 一阶差分
    msrccs_delta = librosa.feature.delta(msrccs, width=3, order=1, axis=- 1, mode='interp')
    # 二阶差分
    msrccs_delta2 = librosa.feature.delta(msrccs, order=2)
    print('N:', N.shape, '|',  'mel:', M.shape, '|', 'mel_d1:', mfcc_delta.shape, '|', 'mel_d2:',
          mfcc_delta2.shape, '|', 'bfccs:', bfccs.shape, '|', 'bfcc_d1:', bfccs_delta.shape, '|',
          'bfcc_d2:', bfccs_delta2.shape, 'msrccs:','|', 'gfccs:', gfccs.shape, '|', 'gfccs_d1', gfccs_delta.shape,
          '|', 'gfccs_d2', gfccs_delta2.shape
          ,'msrccs_d1:', msrccs_delta.shape,
          '|', 'msrccs_d2:', msrccs_delta2.shape)

    LIST = [N.shape[0], M.shape[0], mfcc_delta.shape[0], mfcc_delta2.shape[0], bfccs.shape[0],bfccs_delta.shape[0], bfccs_delta2.shape[0],gfccs.shape[0], gfccs_delta.shape[0], gfccs_delta2.shape[0],msrccs_delta.shape[0], msrccs_delta2.shape[0]]
    # 统一帧长
    frame = np.min(LIST)
    frame = int(frame)

    l_M = list(M)
    l_mfcc_delta = list(mfcc_delta)
    l_mfcc_delta2 = list(mfcc_delta2)

    l_bfccs = list(bfccs)
    l_bfccs_delta = list(bfccs_delta)
    l_bfccs_delta2 = list(bfccs_delta2)

    i_gfccs = list(gfccs)
    l_gfccs_delta = list(gfccs_delta)
    l_gfccs_delta2 = list(gfccs_delta2)
    l_msrccs = list(msrccs)
    l_msrccs_delta = list(msrccs_delta)
    l_msrccs_delta2 = list(msrccs_delta2)

    if  int(len(l_M)) != frame or int(len(l_mfcc_delta)) != frame or \
        int(len(l_mfcc_delta2)) != frame  or int(len(l_bfccs)) != frame or int(len(l_bfccs_delta)) != frame or int(len(l_bfccs_delta2)) != frame \
            or int(len(i_gfccs)) != frame or int(len(l_gfccs_delta)) != frame \
            or int(len(l_gfccs_delta2)) !=frame or int(len(l_msrccs)) != frame or int(len(l_msrccs_delta)) != frame \
            or int(len(l_msrccs_delta2)) != frame:


        N_M = M[0:frame]
        N_M = np.array(N_M)

        N_mfcc_delta = mfcc_delta[0:frame]
        N_mfcc_delta = np.array(N_mfcc_delta)

        N_mfcc_delta2 = mfcc_delta2[0:frame]
        N_mfcc_delta2 = np.array(N_mfcc_delta2)

        N_bfccs = bfccs[0:frame]
        N_bfccs = np.array(N_bfccs)

        N_bfccs_delta = bfccs_delta[0:frame]
        N_bfccs_delta = np.array(N_bfccs_delta)

        N_bfccs_delta2 = bfccs_delta2[0:frame]
        N_bfccs_delta2 = np.array(N_bfccs_delta2)

        # i_gfccs = list(gfccs)
        # l_gfccs_delta = list(gfccs_delta)
        # l_gfccs_delta2 = list(gfccs_delta2)
        N_gfccs = gfccs[0:frame]
        N_gfccs = np.array(N_gfccs)

        N_gfccs_delta = gfccs_delta[0:frame]
        N_gfccs_delta = np.array(N_gfccs_delta)

        N_gfccs_delta2 = gfccs_delta2[0:frame]
        N_gfccs_delta2 = np.array(N_gfccs_delta2)
        N_msrccs = msrccs[0:frame]
        N_msrccs = np.array(N_msrccs)

        N_msrccs_delta = msrccs_delta[0:frame]
        N_msrccs_delta = np.array(N_msrccs_delta)

        N_msrccs_delta2 = msrccs_delta2[0:frame]
        N_msrccs_delta2 = np.array(N_msrccs_delta2)

        mix_feature = np.concatenate((N_M, N_mfcc_delta, N_mfcc_delta2, N_bfccs, N_bfccs_delta,
                                      N_bfccs_delta2,N_gfccs, N_gfccs_delta,
                                      N_gfccs_delta2,  N_msrccs, N_msrccs_delta, N_msrccs_delta2), axis=1)
    else:
        mix_feature = np.concatenate((M, mfcc_delta, mfcc_delta2, bfccs, bfccs_delta, bfccs_delta2,gfccs, gfccs_delta, gfccs_delta2
                                        ,msrccs, msrccs_delta, msrccs_delta2
                                      ), axis=1)

    print('mix_feature:', mix_feature.shape)

    return mix_feature, frame


def truelabel(txt_path, txtfile, frame, inc, wlen):
    '''
    :param txt_path: 真实标注值txt文件存放路径
    :param txtfile:  真实标注值txt文件存放的列表
    :param frame: 帧数
    :return:
    '''
    txt_contest = np.loadtxt(os.path.join(txt_path, txtfile))
    true_label = [0] * frame  # 帧数
    # print(file_test)
    # ---------------平铺列表-----------------------------------------
    b = txt_contest.flatten()  # 存放语音所在的时间位置
    # print("列表长度:", b)
    # --------------------------------------------------------------
    space = []  # 定义一个临时空间
    j = 0
    while j <= len(b) - 1:  # 定义一个j，最大不超过列表b的长度
        space.append(b[j])
        space.append(b[j + 1])  # 取出列表里的第一对值放到临时空间space
        # print(space)
        k1 = math.floor((space[0] * 16000 - wlen) / inc + 1)  # 定义k为space中的第一个值
        # k1 = int(k1 * c.num)
        k2 = math.floor((space[1] * 16000 - wlen) / inc + 1)
        # k2 = int(k2 * c.num)
        # print("第{}个值".format(j), len(a))
        while k1 <= k2:
            true_label[int(k1)] = 1  # 更新列表对应位置a的值
            k1 += 1
        space.clear()  # 清空空间
        j += 2  # 取列表的第二对值
    print('标签:', len(true_label))
    return true_label


def read_data(PATH, name_f, name_l, save_path, name):
    """
    :param PATH: train_fea.csv/train_label所在路径
    :param name_f: 特征文件名
    :param name_l: 标签文件名
    :param save_path: 存储路径
    :param name: 存储的文件名
    :return:
    """
    F = np.loadtxt(open(PATH + name_f, "rb"), delimiter=",", skiprows=0)  # CSV文件转化为数组
    # F = scipy.io.loadmat(os.path.join(c.datafile, name_f))
    # F = F[key_f]
    print("特征维度：", F.shape)
    L = np.loadtxt(open(PATH + name_l, "rb"), delimiter=",", skiprows=0)  # CSV文件转化为数组
    L = np.expand_dims(L, axis=1)
    # L = scipy.io.loadmat(os.path.join(c.datafile, name_l))
    # L = L[key_l]
    print("标签维度：", L.shape)
    mix = np.concatenate((F, L), axis=1)
    print("合并后的维度:", mix.shape)
    np.savetxt(save_path + name + '.csv', mix, fmt='%10.5f', delimiter=',')
    # scipy.io.savemat(save_path + name + '.mat', {'data': mix})


# # ******************************************TRAIN FILE********************************************
# TRAIN_wavfile = file_name(F.TRAIN_WAV_DIR, postfix='.wav')
# TRAIN_filename = get_file_name(F.TRAIN_WAV_DIR)
# TRAIN_txtfile = file_name(F.TRAIN_TXT_DIR, postfix='.txt')
#
# for i in range(len(TRAIN_txtfile)):
#     print('*******************', '文件{}:'.format(i + 1), TRAIN_wavfile[i], '*************************')
#     sr = 44100
#     wlen = 1024
#     inc = 512
#
#     sig, fs = librosa.load(os.path.join(F.TRAIN_WAV_DIR, TRAIN_wavfile[i]), sr=None)
#     sig = librosa.resample(sig, fs, 44100)
#
#     mix_feature, frame = compute_features(sig, sr, inc, wlen)
#     np.savetxt(F.TRAIN_FEAT_DIR + TRAIN_filename[i] + '.csv', mix_feature, fmt='%10.5f', delimiter=',')  # 存储特征
#
#     true_label = truelabel(F.TRAIN_TXT_DIR, TRAIN_txtfile[i], frame, inc, wlen)  # 真实标签
#     np.savetxt(F.TRAIN_LABEL_DIR + TRAIN_filename[i] + '.csv', true_label, fmt='%10.5f', delimiter=',')  # 存储标签
#
# # 合并文件
# train_lab_file = file_name(F.TRAIN_LABEL_DIR, postfix='.csv')  # 将生成的单个标签文件放入train_lab_file列表
# train_fea_file = file_name(F.TRAIN_FEAT_DIR, postfix='.csv')  # 将生成的单个特征文件放入train_fea_file列表
#
# TRAIN = o.DATA(label_file_list=train_lab_file, feature_file_list=train_fea_file)  # 加载需要处理的特征和标签// 类实例化
# TRAIN.CONCATENATE(lab_path=F.TRAIN_LABEL_DIR, fea_path=F.TRAIN_FEAT_DIR,  # 对多个文件进行整合处理，将fea、lab分别统一成一个文件
#                   data_save=F.datafile, name1="train_lab", name2="train_fea")
#
# # 将特征、标签整理成一个文件
# read_data(PATH=F.datafile, name_f="train_fea.csv", name_l="train_lab.csv", save_path=F.csv_data_path, name="train")
#
# # *******************************************VAL FILE*********************************************
# VAL_wavfile = file_name(F.VAL_WAV_DIR, postfix='.wav')
# VAL_filename = get_file_name(F.VAL_WAV_DIR)
# VAL_txtfile = file_name(F.VAL_TXT_DIR, postfix='.txt')
#
# for i in range(len(VAL_txtfile)):
#     print('*******************', '文件{}:'.format(i + 1), VAL_wavfile[i], '*************************')
#     sr = 44100
#     wlen = 1024
#     inc = 512
#
#     sig, fs = librosa.load(os.path.join(F.VAL_WAV_DIR, VAL_wavfile[i]), sr=None)
#     sig = librosa.resample(sig, fs, 44100)
#
#     mix_feature, frame = compute_features(sig, sr, inc, wlen)
#     np.savetxt(F.VAL_FEAT_DIR + VAL_filename[i] + '.csv', mix_feature, fmt='%10.5f', delimiter=',')
#
#     true_label = truelabel(F.VAL_TXT_DIR, VAL_txtfile[i], frame, inc, wlen)  # 真实标签
#     np.savetxt(F.VAL_LABEL_DIR + VAL_filename[i] + '.csv', true_label, fmt='%10.5f', delimiter=',')
#
# val_lab_file = file_name(F.VAL_LABEL_DIR, postfix='.csv')  # 将生成的单个标签文件放入train_lab_file列表
# val_fea_file = file_name(F.VAL_FEAT_DIR, postfix='.csv')  # 将生成的单个特征文件放入train_fea_file列表
#
# VAL = o.DATA(label_file_list=val_lab_file, feature_file_list=val_fea_file)  # 加载需要处理的特征和标签// 类实例化
# VAL.CONCATENATE(lab_path=F.VAL_LABEL_DIR, fea_path=F.VAL_FEAT_DIR,  # 对多个文件进行整合处理，将fea、lab分别统一成一个文件
#                 data_save=F.datafile, name1="val_lab", name2="val_fea")
#
# # 将特征、标签整理成一个文件
# read_data(PATH=F.datafile, name_f="val_fea.csv", name_l="val_lab.csv", save_path=F.csv_data_path, name="val")

# # **********************************************TEST FILE**********************************************
TEST_wavfile = file_name(F.TEST_WAV_DIR, postfix='.wav')
TEST_filename = get_file_name(F.TEST_WAV_DIR)
TEST_txtfile = file_name(F.TEST_TXT_DIR, postfix='.txt')

for i in range(len(TEST_wavfile)):
    print('*******************', '文件{}:'.format(i + 1), TEST_wavfile[i], '*************************')
    sr = 16000
    wlen = 1024
    inc = 256

    sig, fs = librosa.load(os.path.join(F.TEST_WAV_DIR, TEST_wavfile[i]), sr=None)
    # sig = librosa.resample(sig, fs, 44100)

    mix_feature, frame = compute_features(sig, sr, inc, wlen)
    np.savetxt(F.TEST_FEAT_DIR + TEST_filename[i] + '.csv', mix_feature, fmt='%10.5f', delimiter=',')

    # true_label = truelabel(F.TEST_TXT_DIR, TEST_txtfile[i], frame, inc, wlen)  # 真实标签
    # np.savetxt(F.TEST_LABEL_DIR + TEST_filename[i] + '.csv', true_label, fmt='%10.5f', delimiter=',')

# test_lab_file = file_name(F.TEST_LABEL_DIR, postfix='.csv')  # 将生成的单个标签文件放入train_lab_file列表
# test_fea_file = file_name(F.TEST_FEAT_DIR, postfix='.csv')  # 将生成的单个特征文件放入train_fea_file列表

# TEST = o.DATA(label_file_list=test_lab_file, feature_file_list=test_fea_file)  # 加载需要处理的特征和标签// 类实例化
#
# TEST.CONCATENATE(lab_path=F.TEST_LABEL_DIR, fea_path=F.TEST_FEAT_DIR,  # 对多个文件进行整合处理，将fea、lab分别统一成一个文件
#                  data_save=F.datafile, name1="test_lab", name2="test_fea")

# # 将特征、标签整理成一个文件
# read_data(PATH=F.datafile, name_f="test_fea.csv", name_l="test_lab.csv", save_path=F.csv_data_path, name="test")
