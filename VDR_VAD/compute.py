import wave
import contextlib
import os
from glob import glob


def find_files(directory, pattern='**/*.wav'):  # 此处pattern='**/*.wav'是默认参数，当没有给函数传递pattern参数时，就是用默认的
    """Recursively finds all files matching the pattern.
    函数功能：匹配所有的符合条件的文件，并将其以list的形式返回。
    directory='audio/LibriSpeechSamples/train-clean-100-npy/'找出该文件夹下以.wav为后缀的文件，并将文件地址存入了list中
     recursive=True 此处的作用应该类似于for循环 遍历一遍该文件夹
    """
    return glob(os.path.join(directory, pattern), recursive=True)


# 能将wav_path 目录下的所有wav文件的时长统计出来
wav_path = '/usr/local/pyprograme/dh/speech_processing/new_speech/val_wavnew'

file_wav = find_files(wav_path)

total_time = 0
for file in file_wav:
    with contextlib.closing(wave.open(file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        total_time = total_time + duration
        print("duration = ", duration)
        print("total_time = ", total_time)
print("总的语音段数 = ", len(file_wav))
