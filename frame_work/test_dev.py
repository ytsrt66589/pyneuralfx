import torch
from dataset import Full_Modeling_AudioDataset, SnapShot_AudioDataset


TRAIN_PATH = '/home/yytung/projects/pyneuralfx/frame_work/data/overdrive/boss_od3'
PRE_ROOM = 0
BUFFER_SIZE = 2048
NORM_TENSOR = [
    [0, 4],
    [0, 4]
]
SR = 48000 
NUM_CONDS = 2 
BATCH_SIZE = 16 

'''
train_set = Full_Modeling_AudioDataset(
    TRAIN_PATH, 
    pre_room=PRE_ROOM,
    win_len=BUFFER_SIZE, 
    norm_tensor=NORM_TENSOR,
    sr=SR,
    cond_size=NUM_CONDS
)
'''

train_set = SnapShot_AudioDataset(
    input_data_path = '/home/yytung/projects/pyneuralfx/frame_work/data/overdrive/boss_od3/x/x_d0_t2.wav', 
    target_data_path = '/home/yytung/projects/pyneuralfx/frame_work/data/overdrive/boss_od3/y/y_d0_t2.wav', 
    sr = SR, 
    buffer_size = BUFFER_SIZE, 
    pre_room = PRE_ROOM
)


loader_train = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


for batch in loader_train:
    #x, y, c = batch
    x, y = batch
    print('> x: ', x.shape)
    print('> y: ', y.shape)
    #print('> c: ', c.shape)
    break