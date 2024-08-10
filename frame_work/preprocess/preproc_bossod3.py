import os
import soundfile as sf
import librosa 
import random
from utils import traverse_dir


if __name__ == "__main__":

    print('> ================= Start Preprocessing =================  <')

    train_ratio = 0.6
    valid_ratio = 0.1
    test_ratio = 0.3

    assert (train_ratio + valid_ratio + test_ratio) == 1
    
    path_dir_x = './data/overdrive/boss_od3/x'
    path_dir_y = './data/overdrive/boss_od3/x'

    base_dir = './data/overdrive/boss_od3'

    os.makedirs(base_dir, exist_ok=True)

    train_path_dir_x = os.path.join(base_dir, 'train', 'x')
    train_path_dir_y = os.path.join(base_dir, 'train', 'y')
    os.makedirs(train_path_dir_x, exist_ok=True)
    os.makedirs(train_path_dir_y, exist_ok=True)

    valid_path_dir_x = os.path.join(base_dir, 'valid', 'x')
    valid_path_dir_y = os.path.join(base_dir, 'valid', 'y')
    os.makedirs(valid_path_dir_x, exist_ok=True)
    os.makedirs(valid_path_dir_y, exist_ok=True)

    test_path_dir_x = os.path.join(base_dir, 'test', 'x')
    test_path_dir_y = os.path.join(base_dir, 'test', 'y')
    os.makedirs(test_path_dir_x, exist_ok=True)
    os.makedirs(test_path_dir_y, exist_ok=True)

    filelist_x = traverse_dir(path_dir_x, is_pure=True, is_sort=True)
    filelist_y = traverse_dir(path_dir_y, is_pure=True, is_sort=True)

    for i, (fn_x, fn_y) in enumerate (zip(filelist_x, filelist_y)):

        print('> =============================== <')
        print('fn_x: ', fn_x)
        path_x = os.path.join(path_dir_x, fn_x)

        print('fn_y: ', fn_y)
        path_y = os.path.join(path_dir_y, fn_y)
        print('> =============================== <')
        wav_x, sr_x = sf.read(path_x)
        wav_y, sr_y = sf.read(path_y)

        min_len = min(len(wav_x), len(wav_y))
        wav_x = wav_x[:min_len]
        wav_y = wav_y[:min_len]
        print('> len(wav): ', len(wav_x))
        assert sr_x == sr_y 
        assert len(wav_x) == len(wav_y)

        # train
        train_end = int(len(wav_x) * train_ratio)
        train_wav_x = wav_x[:train_end]
        train_wav_y = wav_y[:train_end]
        sf.write(os.path.join(train_path_dir_x, fn_x), train_wav_x, sr_x, subtype='PCM_24')
        sf.write(os.path.join(train_path_dir_y, fn_y), train_wav_y, sr_y, subtype='PCM_24')

        # valid
        valid_end = int(len(wav_x) * valid_ratio)
        valid_wav_x = wav_x[train_end:valid_end+train_end]
        valid_wav_y = wav_y[train_end:valid_end+train_end]
        sf.write(os.path.join(valid_path_dir_x, fn_x), valid_wav_x, sr_x, subtype='PCM_24')
        sf.write(os.path.join(valid_path_dir_y, fn_y), valid_wav_y, sr_y, subtype='PCM_24')

        # test
        test_end = int(len(wav_x) * test_ratio)
        test_wav_x = wav_x[valid_end+train_end:]
        test_wav_y = wav_y[valid_end+train_end:]
        sf.write(os.path.join(test_path_dir_x, fn_x), test_wav_x, sr_x, subtype='PCM_24')
        sf.write(os.path.join(test_path_dir_y, fn_y), test_wav_y, sr_y, subtype='PCM_24')


