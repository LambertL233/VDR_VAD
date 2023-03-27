import os

# Wave path
TRAIN_WAV_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/train_wavnew'
TEST_WAV_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/test_wavxiugaigaoyong'
VAL_WAV_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/val_wavnew'

# Txt path
TRAIN_TXT_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/train_txtnew'
TEST_TXT_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/test_txtxiugaigaoyong'
VAL_TXT_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/val_txtnew'

# Feature path
TRAIN_FEAT_DIR = '/usr/local/pyprograme/dh/mixfeature/train_feature/'
TEST_FEAT_DIR = '/usr/local/pyprograme/dh/SE/test_feature/'
# TEST_FEAT_DIR = '/usr/local/pyprograme/dh/1115/test_fea/'
VAL_FEAT_DIR = '/usr/local/pyprograme/dh/mixfeature/val_feature/'

# Label path
TRAIN_LABEL_DIR = '/usr/local/pyprograme/dh/mixfeature/train_label/'
TEST_LABEL_DIR = '/usr/local/pyprograme/dh/mixfeature/test_label/'
VAL_LABEL_DIR = '/usr/local/pyprograme/dh/mixfeature/val_label/'

# train data\test data
datafile = '/usr/local/pyprograme/dh/mixfeature/fea'  # 文件存储路径

csv_data_path = '/usr/local/pyprograme/dh/mixfeature/csv/'

# Settings for feature extraction
# SAMPLE_RATE = 44100
# WLEN = 2048
# INC = int(WLEN - round(0.01 * SAMPLE_RATE))
# num = float(format(441 / 256, '.4f'))

