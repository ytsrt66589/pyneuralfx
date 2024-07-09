import torch 
from pyneuralfx.models.cnn.tcn import *
from pyneuralfx.models.cnn.gcn import *

B = 32
INP = 1 
OUTP = 1 
N_CONDS = 2
LENGTH = 8192 
SR = 48000

x = torch.randn(B, INP, LENGTH )
c = torch.randn(B, N_CONDS)
h = None


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



'''
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


