import torch 
from pyneuralfx.models.cnn.tcn import *
from pyneuralfx.models.cnn.gcn import *
from pyneuralfx.models.rnn.gru import * 
from pyneuralfx.models.rnn.lstm import * 
from pyneuralfx.models.rnn.rnn import * 

B = 32
INP = 2
OUTP = 2 
N_CONDS = 2
LENGTH = 8192 
SR = 48000
RNN_SIZE = 16 

x = torch.randn(B, INP, LENGTH)
c = torch.randn(B, N_CONDS)
h = torch.zeros(1, B, RNN_SIZE)




print(' > ================ Concat GRU ================ <')
model = ConcatGRU(
    inp_channel = INP,
    out_channel = OUTP,
    rnn_size = RNN_SIZE,
    sample_rate = SR, 
    num_conds = N_CONDS,
    layer_norm = False
)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out, _, _ = model(x, c, h)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ Concat LSTM ================ <')
model = ConcatLSTM(
    inp_channel = INP,
    out_channel = OUTP,
    rnn_size = RNN_SIZE,
    sample_rate = SR, 
    num_conds = N_CONDS,
    layer_norm = False
)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out, _, _ = model(x, c, (h, h))
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ Concat Vanilla RNN ================ <')
model = ConcatVanillaRNN(
    inp_channel = INP,
    out_channel = OUTP,
    rnn_size = RNN_SIZE,
    sample_rate = SR, 
    num_conds = N_CONDS,
    layer_norm = False
)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out, _, _ = model(x, c, h)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


'''
print(' > ================ StaticHyper GRU ================ <')
model = StaticHyperGRU(
    inp_channel = INP,
    out_channel = OUTP,
    rnn_size = RNN_SIZE,
    sample_rate = SR, 
    num_conds = N_CONDS,
    layer_norm = True
)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out, _, _ = model(x, c, h)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ StaticHyper LSTM ================ <')
model = StaticHyperLSTM(
    inp_channel = INP,
    out_channel = OUTP,
    rnn_size = RNN_SIZE,
    sample_rate = SR, 
    num_conds = N_CONDS,
    layer_norm = True
)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out, _, _ = model(x, c, (h, h))
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ StaticHyper Vanilla RNN ================ <')
model = StaticHyperVanillaRNN(
    inp_channel = INP,
    out_channel = OUTP,
    rnn_size = RNN_SIZE,
    sample_rate = SR, 
    num_conds = N_CONDS,
    layer_norm = True
)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out, _, _ = model(x, c, h)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')
'''





'''
print(' > ================ Snapshot TCN ================ <')
model = SnapshotTCN(INP, OUTP, N_CONDS, SR)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out = model(x, c)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ Snapshot Analog TCN ================ <')
model = SnapshotAnalogTCN(INP, OUTP, N_CONDS, SR)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out = model(x, c)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ Concat TCN ================ <')
model = ConcatTCN(INP, OUTP, N_CONDS, SR)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out = model(x, c)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ FiLM TCN ================ <')
model = FiLMTCN(INP, OUTP, N_CONDS, SR)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out = model(x, c)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ Hyper TCN ================ <')
model = HyperTCN(INP, OUTP, N_CONDS, SR)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out = model(x, c)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ Snapshot GCN ================ <')
model = SnapShotGCN(INP, OUTP, N_CONDS, SR)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out = model(x, c)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ Snapshot Analog GCN ================ <')
model = SnapShotAnalogGCN(INP, OUTP, N_CONDS, SR)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out = model(x, c)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')

print(' > ================ Concat GCN ================ <')
model = ConcatGCN(INP, OUTP, N_CONDS, SR)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out = model(x, c)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ FiLM GCN ================ <')
model = FiLMGCN(INP, OUTP, N_CONDS, SR)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out = model(x, c)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')


print(' > ================ Hyper GCN ================ <')
model = HyperGCN(INP, OUTP, N_CONDS, SR)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out = model(x, c)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')

'''

