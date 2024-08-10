import os
import soundfile as sf
import librosa 
import random
from utils import traverse_dir



if __name__ == "__main__":
    print('> ================= Start Preprocessing =================  <')

    # ==================================================== #
    # configuration
    # ==================================================== #
    # Determine the train/valid/test ratio
    train_ratio = 0.6
    valid_ratio = 0.1 
    test_ratio = 0.3 

    assert (train_ratio + valid_ratio + test_ratio) == 1 
    
    # Determine the snapshot audio pair 
    path_to_x = './example_wavs/snapshot_examples/input.wav'
    path_to_y = './example_wavs/snapshot_examples/output.wav'

    # Determine where to save the data split to train/valid/test 
    path_dir_to_save = './data/snapshot_modeling_example'


    os.makedirs(path_dir_to_save, exist_ok=True)
    train_dir_to_save = os.path.join(path_dir_to_save, 'train')
    os.makedirs(train_dir_to_save, exist_ok=True)
    valid_dir_to_save = os.path.join(path_dir_to_save, 'valid')
    os.makedirs(valid_dir_to_save, exist_ok=True)
    test_dir_to_save = os.path.join(path_dir_to_save, 'test')
    os.makedirs(test_dir_to_save, exist_ok=True)

    # load data 
    input_x, sr_x = librosa.load(path_to_x, sr=None, mono=True)
    output_y, sr_y = librosa.load(path_to_y, sr=None, mono=True)
    
    assert sr_x == sr_y 

    # ensure the same length of the input-output pair 
    min_len = min(len(input_x), len(output_y))
    wav_x = input_x[:min_len]
    wav_y = output_y[:min_len]
    
    # split training part 
    train_length = int(min_len * train_ratio)
    train_wav_x = input_x[:train_length]
    train_wav_y = output_y[:train_length]
    sf.write(os.path.join(train_dir_to_save, 'train_x.wav'), train_wav_x, sr_x, subtype='PCM_24')
    sf.write(os.path.join(train_dir_to_save, 'train_y.wav'), train_wav_y, sr_y, subtype='PCM_24')
    

    # split validation part 
    valid_length = int(min_len * valid_ratio)
    valid_wav_x = input_x[train_length:train_length+valid_length]
    valid_wav_y = output_y[train_length:train_length+valid_length]
    sf.write(os.path.join(valid_dir_to_save, 'valid_x.wav'), valid_wav_x, sr_x, subtype='PCM_24')
    sf.write(os.path.join(valid_dir_to_save, 'valid_y.wav'), valid_wav_y, sr_y, subtype='PCM_24')


    # split test part 
    test_length = int(min_len * train_ratio)
    test_wav_x = input_x[train_length+valid_length:]
    test_wav_y = output_y[train_length+valid_length:]
    sf.write(os.path.join(test_dir_to_save, 'test_x.wav'), test_wav_x, sr_x, subtype='PCM_24')
    sf.write(os.path.join(test_dir_to_save, 'test_y.wav'), test_wav_y, sr_y, subtype='PCM_24')


    print('> ================= Finish Preprocessing =================  <')