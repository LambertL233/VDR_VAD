# _*_ coding:utf-8 _*_
import os

from tensorflow.keras.callbacks import LearningRateScheduler
import librosa
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import GRU
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from mixfeature11 import configure as c
from tensorflow.keras.layers import LSTM
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Activation, Dense,Convolution2D, MaxPooling2D
from  tensorflow.keras import backend as K
from tensorflow.keras import backend as K
# K.set_image_dim_ordering('th')
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,Reshape,Add,dot
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Lambda,RepeatVector,multiply,Permute,Concatenate
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import ReLU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用单块GPU，指定其编号即可 （0 or 1or 2 or 3）
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'  # 使用多块GPU，指定其编号即可  (引号中指定即可)
import tensorflow.keras.layers as KL
import librosa.display

# 得到wav文件列表
def file_name(path, postfix='.txt'):
    F = []
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            if os.path.splitext(file)[1] == postfix:
                F.append(file)  # 将所有的文件名添加到F列表中
    return F  # 返回F列表


# 标准化
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def preprare_features_and_labels(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y




# L2正则化
# def L2_W(w):
#     return tf.reduce_sum((w ** 2)) / 2


# 模型定义
# def Model(batchsize, input_length):
#     model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(1, 121)))
    # tf.keras.layers.Dropout(0.4)

    # model.add(tf.keras.layers.Dense(150, input_shape=(batchsize, input_length), activation='relu'))  # 添加层
    # tf.keras.layers.Dropout(0.5)
    # model.add(tf.keras.layers.Dense(30, activation='relu'))
    # tf.keras.layers.Dropout(0.5)
    # model.add(tf.keras.layers.Dense(50, activation='relu'))
    # tf.keras.layers.Dropout(0.5)
    # model.add(tf.keras.layers.Dense(10, activation='relu'))
    # tf.keras.layers.Dropout(0.5)
    # model.add(tf.keras.layers.LSTM(32, activation='relu',return_sequences=True))  # 添加层
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # model.summary()
    # 保存模型结构和参数到文件
    # tf.keras.models.save_model(model, filepath='/home/dirtychicken/桌面/dh')  # 默认生成 .pb 格式模型，也可以通过save_format 设置 .h5 格式
    # print('模型已保存')




# 评估标准
def average(A):
    sum1 = 0.0  # loss
    sum2 = 0.0  # acc
    sum3 = 0.0  # precision
    sum4 = 0.0  # recall
    sum5 = 0.0  # AUC
    for i in range(len(A)):
        sum1 = sum1 + float(A[i][0])
        sum2 = sum2 + float(A[i][1])
        sum3 = sum3 + float(A[i][2])
        sum4 = sum4 + float(A[i][3])
        sum5 = sum5 + float(A[i][4])
    average_loss = sum1 / len(A)
    average_acc = sum2 / len(A)
    average_precision = sum3 / len(A)
    average_recall = sum4 / len(A)
    average_AUC = sum5 / len(A)
    return print('average_loss:', average_loss, 'average_acc:', average_acc, 'average_precision:', average_precision,
                 'average_recall:', average_recall, 'average_AUC:', average_AUC)
    # return print('average_loss:', average_loss, 'average_acc:', average_acc, 'average_AUC:', average_AUC)


# 可视化
def visual(pred, true):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(pred, color='grey', LineWidth=0.5, label='pred')
    ax.plot(true, color='black', LineWidth=1.0, label='true')

    plt.xlabel('time/s')
    plt.ylabel('class')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #     plt.legend()


def Fmeasure(a, A):
    '''
    :param P: Precision
    :param R: Recall
    :param a: parameter
    :return:
    '''
    sum6 = 0.0  # Fmeature
    # A[i][2]指的Precision，A[i][3]指的Recall
    for i in range(len(A)):
        Fmeasure = (float(a) ** 2 + 1.0) * A[i][2] * A[i][3] / (float(a) ** 2 * A[i][2] + A[i][3])
        sum6 = sum6 + Fmeasure
        print('第{}个文件：'.format(i), 'Precision:', A[i][2], 'Recall:', A[i][3], 'Fmeasure:', Fmeasure)
    average_Fmeasure = sum6 / len(A)
    return print('average_Fmeasure:', average_Fmeasure)

lr = 0.001
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch %20 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.95)
        print("lr changed to {}".format(lr * 0.95))
    return K.get_value(model.optimizer.lr)


# ---------------------------------------------数据加载------------------------------------------------------------------
test_file = file_name(c.TEST_FEAT_DIR, postfix='.csv')
test_lab_file = file_name(c.TEST_LABEL_DIR, postfix='.csv')

val_file = file_name(c.VAL_FEAT_DIR, postfix='.csv')
val_lab_file = file_name(c.VAL_LABEL_DIR, postfix='.csv')

data = pd.read_csv(os.path.join(c.csv_data_path, "train.csv"))
val = pd.read_csv(os.path.join(c.csv_data_path, "val.csv"))

shuffle_data = data.sample(frac=1).reset_index(drop=True)  # 打乱之后，索引按照正常顺序排列
# val_shuffle_data = val.sample(frac=1).reset_index(drop=True)

train_x = shuffle_data.iloc[:, :-1].values  # 特征
train_x = normalize(train_x)  # 归一化
train_y = shuffle_data.iloc[:, -1:].values  # 标签

# test_wav = file_name("/usr/local/pyprograme/dh/speech_processing/new_speech/test1_wav", postfix='.wav')

#计算训练集标签中1和0的数目
l = list(train_y)
zero = l.count(0)
one = l.count(1)
percent = zero / len(l)
print('zero:', zero, 'one:', one, percent)

print(train_x.shape, type(train_x), train_y.shape, type(train_y))

val_x = val.iloc[:, :-1]
val_x = normalize(val_x)
val_x=val_x.values.reshape(-1,20,12,1)
val_y = val.iloc[:, -1:]

# train_x = K.reshape(train_x.shape[0],1)
# val_x =K.reshape(val_x.shape[0],1)
#
# train_y = np.array(tf.keras.utils.to_categorical(train_y, 10))
# val_y = np.array(tf.keras.utils.to_categorical(val_y, 10))
print(train_x.shape)
print(train_y.shape)

train_x=train_x.reshape(-1,20,12,1)


# ----------------------------------------------------------------------------------------------------------------------

# 参数设置
batch_size = 64


Loss = []

num_epoch =60
Train_loss = []
Val_loss = []

index_start = 0
OUTPUT_SIZE = 1
CELL_SIZE = 200
LR = 1e-3
# print('----------->batch_size|', BATCH_SIZE, '---------------->num_epoch|', num_epoch,
#       '---------------------------->learning rate|', LR)


print('build model……')
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(518769, 156, input_length=156))  # 此处必须有input_length,否者无法降维进行全连接
# model.add(tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.5)))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Flatten())
# model.add(Dense(1, activation="sigmoid"))

# model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu',input_shape = (26,6,1),padding = 'same' ,kernel_regularizer=regularizers.l2(0.01),
#                 activity_regularizer=regularizers.l1(0.01)))
#
#
# model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))
# model.add(tf.keras.layers.Convolution2D(64, (3, 3),activation='relu'))
#
# model.add(tf.keras.layers.Convolution2D(128, (3, 3),activation='relu'))
# model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# this converts ours 3D feature maps to 1D feature vector
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))  # 几个分类就几个dense
# model.add(Activation('sigmoid'))  # 多分类
# model.summary()
channel_axis = 1 if K.image_data_format() == "channels_first" else 3

# CAM
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu',kernel_initializer= glorot_uniform(seed = 0), kernel_regularizer=l1_l2(0.02),)(x)
    # x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu',
    #            kernel_initializer=glorot_uniform(seed=0), )(x)

    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)(x)

    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')

    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    x = cbam_module(x)
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=(1,1),padding='same')
        x = tf.keras.layers.add([x, shortcut])
        return x
    else:
        x = tf.keras.layers.add([x, inpt])
        return x


# def Model():
    # model=tf.keras.Sequential()
    # model.add(tf.keras.layers.ZeroPadding2D((3, 3),input_shape=(12,20,1)))
inpt = Input(shape=(20, 12, 1))
x = tf.keras.layers.ZeroPadding2D((3, 3))(inpt)
x = Conv2d_BN(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)



x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
    # model.add(Conv2d_BN(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), padding='valid'))


x= Dense(128)(x)
x=Activation(ReLU(threshold=0.06))(x)
x=Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(1, activation='sigmoid',)(x)

model = Model(inputs=inpt, outputs=x)
# model = Model(inputs=inpt, outputs=x)
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

# tf.keras.models.save_model(model, filepath='/home/dirtychicken/桌面/dh/resnet')  # 默认生成 .pb 格式模型，也可以通过save_format 设置 .h5 格式
# print('模型已保存')






# INPUT_SIZE = 150
BATCH_SIZE = 128
# model = Model(INPUT_SIZE, BATCH_SIZE)
# model.summary()






print("******************************************trian**************************************************************")

print('----------->batch_size|', batch_size, '---------------->num_epoch|', num_epoch,
      '---------------------------->learning rate|', lr)

model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=lr, decay=0.0099),
              metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(),
                       tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives()])
reduce_lr = LearningRateScheduler(scheduler)

history = model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epoch,
                    shuffle=True, validation_data=(val_x, val_y),callbacks=[reduce_lr])
model.save('/home/dirtychicken/桌面/dh/resnet/CBAM2.pb')  # 默认生成 .pb 格式模型，也可以通过save_format 设置 .h5 格式
print('模型已保存')
train_loss = history.history['loss']
val_loss = history.history['val_loss']
Loss.append(train_loss)
Loss = np.array(Loss).reshape(-1, 1)
Val_loss.append(val_loss)
Val_loss = np.array(Val_loss).reshape(-1, 1)
ls = np.concatenate([Loss, Val_loss],  axis=1)

print(Loss.shape)



acc = history.history['acc']
val_acc = history.history['val_acc']
print("*************************************************test*****************************************************")
A = []  # accuracy
for i in range(len(test_file)):
    print('正在测试第{}个文件'.format(i))
    # 加载测试集
    test_data = pd.read_csv(os.path.join(c.TEST_FEAT_DIR, test_file[i]))
    test_lab = pd.read_csv(os.path.join(c.TEST_LABEL_DIR, test_lab_file[i]))

    test_x = test_data.iloc[:, :].values

    test_x = normalize(test_x)
    test_x = test_x.reshape(-1, 12,20, 1)
    # test_x = K.as_matrix().reshape(-1, 26, 6, 1)
    test_y = test_lab.iloc[:, :].values

    # plt.plot(test_y)  # 标签

    # test_x = test_x.reshape(test_x.shape[0], 1)
    # test_y = np.array(tf.keras.utils.to_categorical(test_y, 10))
    a = model.evaluate(test_x, test_y, batch_size=batch_size)  # 用上面跑出来的权重结果测试，存放评估结果，包括loss和acc
    Pre = list(model.predict(test_x))  # 预测的概率值，转换为列表形式
    # Pre=np.argmax(Pre,axis=-1)
    plt.plot(Pre)
    plt.show()

    A.append(a)
    print('第{}个文件测试完成……………………………………'.format(i))

print('result:', average(A))


Fmeasure(1, A)
# # FAP(A)
# fig = plt.figure(figsize=[8,4])
plt.ion()
plt.subplot(2, 1, 1)
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.xticks(np.arange(0, 42, step=2))
plt.title('Training and validation loss')
plt.tight_layout()
plt.legend()
plt.show()


plt.subplot(2, 1, 2)
plt.plot(acc, label='Training acc')
plt.plot(val_acc, label='Validation acc')
plt.title('Training and Validation acc')
plt.tight_layout()
plt.legend()
plt.ioff()
# plt.clf()

plt.show()
