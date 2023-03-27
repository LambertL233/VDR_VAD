import os

# Wave path
TRAIN_WAV_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/train_wavPSO'
TEST_WAV_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/test_wavxiugaigaoyong'
VAL_WAV_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/val_wavnew'

# Txt path
TRAIN_TXT_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/train_txtPSO'
TEST_TXT_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/test_txtxiugaigaoyong'
VAL_TXT_DIR = '/usr/local/pyprograme/dh/speech_processing/new_speech/val_txtnew'

# Feature path
TRAIN_FEAT_DIR = '/usr/local/pyprograme/dh/Psofea/train_feature/'
TEST_FEAT_DIR = '/usr/local/pyprograme/dh/Psofea/test_feaxiugaigaoyong/'
VAL_FEAT_DIR = '/usr/local/pyprograme/dh/Psofea/val_feature/'

# # NEW Feature path
# NEW_TRAIN_FEAT_DIR = '/usr/local/pyprograme/dh/wavelet/new_train_feature/'
# NEW_TEST_FEAT_DIR = '/usr/local/pyprograme/dh/wavelet/new_test_feature/'
# NEW_VAL_FEAT_DIR = '/usr/local/pyprograme/dh/wavelet/new_val_feature/'

# Label path
TRAIN_LABEL_DIR = '/usr/local/pyprograme/dh/Psofea/train_label/'
TEST_LABEL_DIR = '/usr/local/pyprograme/dh/Psofea/test_labelxiugaigaoyong/'
VAL_LABEL_DIR = '/usr/local/pyprograme/dh/Psofea/val_label/'
#
# # NEW Label path
# NEW_TRAIN_LABEL_DIR = '/usr/local/pyprograme/dh/wavelet/new_train_label/'
# NEW_TEST_LABEL_DIR = '/usr/local/pyprograme/dh/wavelet/new_test_label/'
# NEW_VAL_LABEL_DIR = '/usr/local/pyprograme/dh/wavelet/new_val_label/'

# train data\test data
datafile = '/usr/local/pyprograme/dh/Psofea/'  # 文件存储路径

csv_data_path = '/usr/local/pyprograme/dh/Psofea/csv/'

# Settings for feature extraction
SAMPLE_RATE = 44100
WLEN = 2048
INC = 441
num = float(format(441 / 256, '.4f'))
