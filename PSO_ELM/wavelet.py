import combine as o
import configure as c
import DataAlignment as d
import MakeData1 as m
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 得到txt文件列表
train_txt = m.file_name(c.TRAIN_TXT_DIR, postfix='.txt')
test_txt = m.file_name(c.TEST_TXT_DIR, postfix='.txt')
val_txt = m.file_name(c.VAL_TXT_DIR, postfix='.txt')

# 得到wav文件列表
train_wav = m.file_name(c.TRAIN_WAV_DIR, postfix='.wav')
test_wav = m.file_name(c.TEST_WAV_DIR, postfix='.wav')
val_wav = m.file_name(c.VAL_WAV_DIR, postfix='.wav')

# 得到除去拓展名之外的文件名列表
train_file_list = m.get_file_name(c.TRAIN_TXT_DIR)
test_file_list = m.get_file_name(c.TEST_TXT_DIR)
val_file_list = m.get_file_name(c.VAL_TXT_DIR)

print("******************************make train data******************************")
# A = m.AUDIOPROCESS(wav_list=train_wav, file_list=train_file_list)
# A.WAVFEATURE(wav_path=c.TRAIN_WAV_DIR, save_path=c.TRAIN_FEAT_DIR)  # 提取音频特征
# A.AUDIOLABEL(txt_path=c.TRAIN_TXT_DIR, txtfile=train_txt, save_path=c.TRAIN_LABEL_DIR)  # 通过txt文件制作标签
#
# train_fea_file = m.file_name(c.TRAIN_FEAT_DIR, postfix='.csv')  # 将特征放入train_fea_file列表
# train_lab_file = m.file_name(c.TRAIN_LABEL_DIR, postfix='.csv')  # 将标签放入train_lab_file列表
#
#
# TRAIN = o.DATA(label_file_list=train_lab_file, feature_file_list=train_fea_file)  # 加载需要处理的新的特征和标签
# TRAIN.CONCATENATE(lab_path=c.TRAIN_LABEL_DIR, fea_path=c.TRAIN_FEAT_DIR,  # 对多个文件进行整合处理
#                   data_save=c.datafile, name1="train_lab", name2="train_fea")

print("****************************make test data*********************************")
B = m.AUDIOPROCESS(wav_list=test_wav, file_list=test_file_list)
B.WAVFEATURE(wav_path=c.TEST_WAV_DIR, save_path=c.TEST_FEAT_DIR)
B.AUDIOLABEL(txt_path=c.TEST_TXT_DIR, txtfile=test_txt, save_path=c.TEST_LABEL_DIR)

test_fea_file = m.file_name(c.TEST_FEAT_DIR, postfix='.csv')
test_lab_file = m.file_name(c.TEST_LABEL_DIR, postfix='.csv')


TEST = o.DATA(label_file_list=test_lab_file, feature_file_list=test_fea_file)
TEST.CONCATENATE(lab_path=c.TEST_LABEL_DIR, fea_path=c.TEST_FEAT_DIR,
                 data_save=c.datafile, name1="test_lab", name2="test_fea")

print("****************************make val data*********************************")
# C = m.AUDIOPROCESS(wav_list=val_wav, file_list=val_file_list)
# C.WAVFEATURE(wav_path=c.VAL_WAV_DIR, save_path=c.VAL_FEAT_DIR)
# C.AUDIOLABEL(txt_path=c.VAL_TXT_DIR, txtfile=val_txt, save_path=c.VAL_LABEL_DIR)
#
# val_fea_file = m.file_name(c.VAL_FEAT_DIR, postfix='.csv')
# val_lab_file = m.file_name(c.VAL_LABEL_DIR, postfix='.csv')
# #
# VAL = o.DATA(label_file_list=val_lab_file, feature_file_list=val_fea_file)
# VAL.CONCATENATE(lab_path=c.VAL_LABEL_DIR, fea_path=c.VAL_FEAT_DIR,
#                 data_save=c.datafile, name1="val_lab", name2="val_fea")

# o.read_data(name_f="train_fea.csv", name_l="train_lab.csv", save_path=c.csv_data_path, name="train")
o.read_data(name_f="test_fea.csv", name_l="test_lab.csv", save_path=c.csv_data_path, name="test")
# o.read_data(name_f="val_fea.csv", name_l="val_lab.csv", save_path=c.csv_data_path, name="val")