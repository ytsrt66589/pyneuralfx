import torch 
from pyneuralfx.models.cnn.tcn import *
from pyneuralfx.models.cnn.gcn import *
from pyneuralfx.models.rnn.gru import * 
from pyneuralfx.models.rnn.lstm import * 
from pyneuralfx.models.rnn.rnn import * 
from pyneuralfx.models.rnn.cells import * 
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
h_hat = torch.zeros(1, B, 8)

#learnable_cell = LearnableCell(input_size = INP , hidden_size = RNN_SIZE , cell_type = 'vanilla_rnn')
#x = x.permute(0, 2, 1)
#out, _ = learnable_cell(x, h)
#out = out.permute(0, 2, 1)
#print('> out shape: ', out.shape)
'''
model = SnapShotGRU(
    inp_channel = INP,
    out_channel = OUTP,
    rnn_size = RNN_SIZE,
    sample_rate = SR, 
    layer_norm = True
)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out, _, _ = model(x, c, h)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')
'''

'''
model = DynamicHyperGRU(
    inp_channel = INP,
    out_channel = OUTP,
    rnn_size = RNN_SIZE,
    sample_rate = SR, 
    num_conds = N_CONDS,
    layer_norm = False
)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out, _, _ = model(x, c, h, h_hat)
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')
'''

'''
model = DynamicHyperLSTM(
    inp_channel = INP,
    out_channel = OUTP,
    rnn_size = RNN_SIZE,
    sample_rate = SR, 
    num_conds = N_CONDS,
    layer_norm = False
)
print('> receptive field: ', model.compute_receptive_field())
print('> num of params: ', model.compute_num_of_params())
out, _, _ = model(x, c, (h, h), (h_hat, h_hat))
print('> out: ', out.shape)
print(' > ============================================ <\n\n\n')
'''
'''
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



print(' > ================ Concat GRU ================ <')
model = ConcatGRU(
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


print(' > ================ Concat LSTM ================ <')
model = ConcatLSTM(
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


print(' > ================ Concat Vanilla RNN ================ <')
model = ConcatVanillaRNN(
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
print(' > ================ FiLMGRU ================ <')
model = FiLMGRU(
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


print(' > ================ FiLM LSTM ================ <')
model = FiLMLSTM(
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


print(' > ================ FiLM Vanilla RNN ================ <')
model = FiLMVanillaRNN(
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



print(' > ================ FiLMGRU ================ <')
model = FiLMGRU(
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


print(' > ================ FiLM LSTM ================ <')
model = FiLMLSTM(
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


print(' > ================ FiLM Vanilla RNN ================ <')
model = FiLMVanillaRNN(
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

