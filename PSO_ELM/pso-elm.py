# libraries
import random
import os
import scipy
import sklearn.metrics
from scipy import linalg
import numpy as np
import pandas as pd
import configure as c

# 得到wav文件列表
def file_name(path, postfix='.txt'):
    F = []
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            if os.path.splitext(file)[1] == postfix:
                F.append(file)  # 将所有的文件添加到F列表中
    return F  # 返回F列表


# data
def Data():
    # membaca data
    # -------------------train data-------------------------
    train_file_name = '/usr/local/pyprograme/dh/Psofea/csv/'
    training = pd.read_csv(os.path.join(train_file_name, "train.csv"), header=None)

    data_training = training.iloc[:, :-1]  # 特征
    train_target = training.iloc[:, -1:]  # 标签

    # mengubah data dalam bentuk matriks
    Data.dt_training = data_training.to_numpy()
    Data.dt_target_training = train_target.to_numpy()

    # --------------------test data--------------------------
    test_feature_file_name ='/usr/local/pyprograme/dh/Psofea/test_feaxiugaigaoyong'
    test_label_file_name = '/usr/local/pyprograme/dh/Psofea/test_labelxiugaigaoyong'
    #
    test_feature = file_name(test_feature_file_name, postfix='.csv')
    test_label = file_name(test_label_file_name, postfix='.csv')
    # test_feature = file_name(c.TEST_FEAT_DIR, postfix='.csv')
    # test_label= file_name(c.TEST_LABEL_DIR, postfix='.csv')
    Data.Feature = []
    Data.Label = []
    for i in range(len(test_feature)):
        testing = pd.read_csv(os.path.join(test_feature_file_name, test_feature[i]), header=None)
        label = pd.read_csv(os.path.join(test_label_file_name, test_label[i]), header=None)

        data_testing = testing.iloc[:, :]  # 特征
        Data.dt_testing = data_testing.to_numpy()

        test_target = label.iloc[:, :]  # 标签
        Data.dt_target_testing = test_target.to_numpy()

        Data.Feature.append(Data.dt_testing)
        Data.Label.append(Data.dt_target_testing)


def Hidden_layer(input_weights, biases, n_hidden_node, data_input):  # 40， 5， 5， Data.dt_training (163256*8)
    # inisialisasi input weight
    input_weight = input_weights.reshape(n_hidden_node, 8)  # 将权重变换为矩阵表示，为了与输入相乘,隐藏层有5个节点 5*8

    # inisialisasi bias
    bias = biases

    # transpose input weight
    transpose_input_weight = np.transpose(input_weight)  # 8*5

    # matriks output hidden layer
    hidden_layer = []  # 存放每一行(相当于每一帧)数据的输出(一行即一帧的输出是1*5)， 将8个特征转换为5个
    for data in range(len(data_input)):  # 数据是163256行，8个特征，这里相当于遍历帧数
        # perkalian data input dengan input weight transpose
        h = np.matmul(data_input[data], transpose_input_weight)  # 计算隐藏层的输出 (163256,8)*(8,5)=(163256,5)

        # penambahan dengan bias
        h_output = np.add(h, bias)  # 将5个偏置项分别与隐藏层5个节点的输出对应相加
        hidden_layer.append(h_output)

    return hidden_layer


# 激活层
def Activation(hidden_layer):
    for row in range(len(hidden_layer)):
        for col in range(len(hidden_layer[row])):
            hidden_layer[row][col] = 1 / (1 + np.exp((hidden_layer[row][col] * (-1))))  # 这里的公式为sigmoid函数
    activation = hidden_layer

    return activation


# 计算伪逆矩阵matriks moore penrose pseudo-inverse menggunakan SVD
def Pseudoinverse(hidden_layer):
    h_pseudo_inverse = scipy.linalg.pinv2(hidden_layer, cond=None, rcond=None, return_rank=False,
                                          check_finite=True)  # 计算伪逆矩阵

    return h_pseudo_inverse


# menghitung output weight
def Output_weight(pseudo_inverse, target):
    beta = np.matmul(pseudo_inverse, target)

    return beta


# menghitung hasil prediksi pada data testing
def Target_output(testing_hidden_layer, output_weight):
    target = np.matmul(testing_hidden_layer, output_weight)
    # memetakan matriks target pada klasifikasi
    prediction = []
    for result in range(len(target)):
        dist_target_0 = abs(target[result] - 0)
        dist_target_1 = abs(target[result] - 1)
        min_dist = min(dist_target_0, dist_target_1)
        if min_dist == dist_target_0:
            predict = 0
        elif min_dist == dist_target_1:
            predict = 1
        prediction.append(predict)

    return prediction


# 初始化微粒/位置 (input weight & bias)
def Particle(n_inputWeights, n_biases):
    # inisialisasi input weight
    input_weights = []
    for input_weight in range(0, n_inputWeights):
        input_weights.append(round(random.uniform(-1.0, 1.0), 2))  # 保留两位小数

    # inisialisasi bias
    biases = []
    for bias in range(0, n_biases):
        biases.append(round(random.random(), 2))  # 保留两位小数

    return input_weights + biases  # 65+5=70


# 速度初始化
def Velocity_0(n_particles):
    return [0] * n_particles


# accuracy
def Evaluate(actual, prediction):
    true = 0
    for i in range(min(len(actual), len(prediction))):
        if actual[i] == prediction[i]:
            true += 1
    # akurasi
    accuracy = round(((true / len(prediction)) * 100), 2)

    return accuracy


# mendapatkan pBest
def Pbest(particles, fitness, F, AUC):
    fitness = np.expand_dims(fitness, axis=1)
    F = np.expand_dims(F, axis=1)
    AUC = np.expand_dims(AUC, axis=1)
    pbest = np.hstack((particles, fitness, F, AUC))  # 对应位置在水平方向上平铺
    '''import numpy as np
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
print np.vstack((arr1,arr2))

print np.hstack((arr1,arr2))

a1=np.array([[1,2],[3,4],[5,6]])
a2=np.array([[7,8],[9,10],[11,12]])
print a1
print a2
print np.hstack((a1,a2))
结果如下：
[[1 2 3]
 [4 5 6]]
[1 2 3 4 5 6]
[[1 2]
 [3 4]
 [5 6]]
[[ 7  8]
 [ 9 10]
 [11 12]]
[[ 1  2  7  8]
 [ 3  4  9 10]
 [ 5  6 11 12]]
    '''

    return pbest


# membandingkan pbest ke t dan pbest ke t+1
def Comparison(pbest_t, pbest_t_1):
    for i in range(min(len(pbest_t), len(pbest_t_1))):
        if pbest_t[i][-3] > pbest_t_1[i][-3]:
            pbest_t_1[i] = pbest_t[i]
        else:
            pbest_t_1[i] = pbest_t_1[i]

    return pbest_t_1


# mendapatkan partikel terbaik dalam suatu populasi
def Gbest(particles, fitness, F, AUC):
    # fitness / akurasi terbaik
    best_fitness = np.amax(fitness)
    # partikel dengan fitness terbaik
    particle = fitness.index(best_fitness)
    F1 = F[particle]
    AUC_1 = AUC[particle]
    best_particle = particles[particle]

    # gbest
    gbest = np.hstack((best_particle, best_fitness, F1, AUC_1))

    return gbest


# update kecepatan  更新速度
def Velocity_update(pbest, gbest, w, c1, c2, particles, velocity):
    # mencari batas tiap fitur
    interval = []
    for i in range(len(particles[0])):
        x_max = np.amax(np.array(particles)[:, i])
        x_min = np.amin(np.array(particles)[:, i])
        k = round(random.random(), 1)
        v_max_i = np.array(((x_max - x_min) / 2) * k)
        v_min_i = np.array(v_max_i * -1)
        intvl = np.hstack((v_min_i, v_max_i))
        interval.append(intvl)

    # update kecepatan
    r1 = round(random.random(), 1)
    r2 = round(random.random(), 1)
    for i in range(min(len(particles), len(velocity), len(pbest), len(gbest))):
        for j in range(min(len(particles[i]) - 1, len(pbest[i]) - 1)):
            velocity[i] = (w * velocity[i]) + (c1 * r1 * (pbest[i][j] - particles[i][j])) + (
                    c2 * r2 * (gbest[i] - particles[i][j]))

    return velocity


# update posisi partikel
def Position_update(current_position, velocity_update):
    for i in range(min(len(current_position), len(velocity_update))):
        for j in range(len(current_position[i])):
            current_position[i][j] = (current_position[i][j] + velocity_update[i])

    return current_position


# fungsi ELM
def Elm(particles, n_input_weights, n_hidden_node):  # 100(100*45)，40， 5
    fitness = []  # 正确率
    PP = []
    RR = []
    AUC = []
    F = []

    for i in range(len(particles)):
        # -----------------training---------------------#
        # input weight
        input_weights = np.array(particles[i][0:n_input_weights])  # 40

        # bias
        biases = np.array(particles[i][n_input_weights:len(particles[i])])

        # menghitung matriks keluaran hidden layer pada data training
        hidden_layer_training = Hidden_layer(input_weights, biases, n_hidden_node,
                                             Data.dt_training)  # 计算隐藏层的输出，输出是一个列表，存放163256帧，每帧为(1*5)的大小

        # aktivasi hasil keluaran hidden layer data training

        activation_training = Activation(hidden_layer_training)  # 激活函数

        # matriks moore penrose  # 计算伪逆矩阵
        pseudo_training = Pseudoinverse(activation_training)
        '''
        由于奇异矩阵和非方矩阵不存在可逆矩阵。
        
        m*n的非方矩阵的逆矩阵叫做伪逆矩阵。

        函数返回一个与A的转置矩阵A' 同型的矩阵X，

        并且满足：AXA=A；XAX=X。
           '''
        # menghitung output weight pada data training
        output_training = Output_weight(pseudo_training, Data.dt_target_training)

        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        # -----------------testing--------------------#
        for j in range(len(Data.Feature)):
            # menghitung matriks keluaran hidden layer pada data testing
            hidden_layer_testing = Hidden_layer(input_weights, biases, n_hidden_node, Data.Feature[j])

            # aktivasi matriks keluaran hidden layer data testing
            activation_testing = Activation(hidden_layer_testing)

            # menghitung hasil prediksi pada data testing
            # prediction = Target_output(hidden_layer_testing, output_training)
            prediction = Target_output(activation_testing, output_training)

            # akurasi
            accuracy = Evaluate(Data.Label[j], prediction)
            # da=Data.Label[j]
            acc2 = sklearn.metrics.accuracy_score(Data.Label[j], prediction)
            P = sklearn.metrics.precision_score(Data.Label[j], prediction)
            R = sklearn.metrics.recall_score(Data.Label[j], prediction)
            FF = 2 * P * R / (P + R)
            y, pred, thresholds = sklearn.metrics.roc_curve(Data.Label[j], prediction)
            auc = sklearn.metrics.auc(y, pred)
            print('第{}组权重'.format(i), '第{}个测试文件的准确率:'.format(j), accuracy, '|', acc2, '|', 'P:', P,
                  '|', 'R:', R, '|', 'F:', '|', FF, 'auc:', auc)
            sum1 += accuracy
            sum2 += P
            sum3 += R
            sum4 += auc
            sum5 += FF
        accuracy = np.around(sum1 / len(Data.Feature), 4)  # 计算平均值
        P = np.around(sum2 / len(Data.Feature), 4)
        R = np.around(sum3 / len(Data.Feature), 4)
        auc = np.around(sum4 / len(Data.Feature), 4)
        F1 = np.around(sum5 / len(Data.Feature), 4)

        print('平均准确率：', accuracy, 'P:', P, 'R:', R, 'auc:', auc, 'F:', F1)

        fitness.append(accuracy)
        PP.append(P)
        RR.append(R)
        AUC.append(auc)
        F.append(F1)
        print('{}'.format(i), '----------------------------------------------------------------end')
    return fitness, PP, RR, AUC, F


def Run():
    # 初始化
    fitures = 8 # 特征
    n_hidden_node = 5  # 隐藏层节点数目
    n_input_weights = n_hidden_node * fitures  # 权重数目
    population = 20  # 生成100组权重/偏置
    max_iter = 3  # 最大重复
    w = 0.5  # 惯性因子
    c1 = 1  # 学习因子 1
    c2 = 1  # 学习因子 2
    print('设置超参数', '|', '特征：', fitures, '|', '隐藏节点目：', n_hidden_node, '|', '权重数目：', n_input_weights, '|',
          '迭代次数', max_iter, '|', '惯性因子w:', w, '|', '学习因子c1:', c1, '|',
          '学习因子c2:', c2, '|', '粒子数目:', population)
    print('处理数据……')
    # data
    Data()
    print('速度初始化……')
    # 速度初始化
    velocity_t = Velocity_0(population)  # 初始化速度为0
    print('位置初始化……')
    # 位置初始化
    particles = []  # 100组粒子，每组粒子包含一组权重 转换为数组的话大小应为为100*45(40个权重，5个偏置(每个隐藏层节点对应一个偏置))
    print('随机初始化微粒/位置……')
    # 初始化粒子数目
    for pop in range(population):
        particle = Particle(n_input_weights, n_hidden_node)
        particles.append(particle)
    a = np.array(particles)
    print(a.shape)
    print('-----------------------------------------------------开始训练-----------------------------------------------')
    # fitness tiap partikel = akurasi elm
    fitness_t, PP, RR, AUC, F = Elm(particles, n_input_weights, n_hidden_node)
    print('找出某个粒子最好位置第i维的值……')
    # inisialisasi Pbest
    pbest_t = Pbest(particles, fitness_t, F, AUC)  # 表示某个粒子最好位置第i维的值
    print('找出整个种群最好位置第i维的值……')
    # inisialisasi Gbest
    gbest_t = Gbest(particles, fitness_t, F, AUC)  # 表示整个种群最好位置第i维的值
    print('----------------------------------------------------开始迭代------------------------------------------------')
    for iteration in range(max_iter):
        # update kecepatan
        print('第{}次迭代'.format(iteration))
        velocity_t_1 = Velocity_update(pbest_t, gbest_t, w, c1, c2, particles, velocity_t)  # 速度更新

        # update posisi partikel
        particles_t_1 = Position_update(particles, velocity_t_1)  # 粒子位置更新

        # elm
        fitness_t_1, PP_1, RR_1, AUC_1, F_1 = Elm(particles_t_1, n_input_weights, n_hidden_node)

        # update pbest
        pbest_t_1 = Pbest(particles_t_1, fitness_t_1, F_1, AUC_1)
        pbest_t_1 = Comparison(pbest_t, pbest_t_1)

        # update gbest
        gbest_t_1 = Gbest(particles_t_1, fitness_t_1, F_1, AUC_1)

        # ------------------------#
        pbest_t = pbest_t_1
        gbest_t = gbest_t_1
        particles = particles_t_1
        velocity_t = velocity_t_1

    print('Input Weights')
    print(gbest_t_1[0:n_input_weights])
    print('')
    print('Biases')
    print(gbest_t_1[n_input_weights:len(gbest_t_1) - 3])
    print('')
    print('Accuracy')
    print(gbest_t_1[-3])
    print('')
    print('F')
    print(gbest_t_1[-2])
    print('')
    print('AUC')
    print(gbest_t_1[-1])


if __name__ == '__main__':
    Run()
