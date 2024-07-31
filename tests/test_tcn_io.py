import torch 
from pyneuralfx import pyneuralfx


from pyneuralfx.models.cnn.tcn import *

B = [1, 32]
INP = [1, 2]
OUTP = [1, 2]
N_CONDS = [1, 2]
LENGTH = [8, 32] 
NUM_BLOCKS = [1, 2]
SR = 48000
RNN_SIZE = [4] 
HYPER_RNN_SIZE = [4]
KERNEL_SIZES = [3, 5]
DILATION_GROWTHS = [2]
NCHANNELS = [8]

for _b in B:
    for _inp in INP:
        for _outp in OUTP:
            for _n in N_CONDS:
                for _l in LENGTH:
                    for _nb in NUM_BLOCKS:
                        for _k in KERNEL_SIZES:
                            for _dl in DILATION_GROWTHS:
                                for _ch in NCHANNELS:
                                    
                                    # output 
                                    expected_output = torch.randn(_b, _outp, _l)
                                    
                                    # snapshot 
                                    model = SnapShotTCN(
                                        n_inp = _inp,
                                        n_output = _outp,
                                        sample_rate = SR,
                                        n_blocks = _nb,
                                        kernel_size = _k, 
                                        dilation_growth = _dl, 
                                        n_channels = _ch
                                    )

                                    _rec_in_samples, _rec_in_sec = model.compute_receptive_field()
                                    
                                    # input 
                                    x = torch.randn(_b, _inp, _l+_rec_in_samples-1)
                                    c = torch.randn(_b, _n)
                                    h = torch.zeros(1, _b, 8)

                                    # 
                                    out = model(x, c, h)
                                    assert out.shape == expected_output.shape

                                    # concat gcn 
                                    model = ConcatTCN(
                                        n_inp = _inp,
                                        n_output = _outp,
                                        sample_rate = SR,
                                        n_cond = _n, 
                                        n_blocks = _nb,
                                        kernel_size = _k, 
                                        dilation_growth = _dl, 
                                        n_channels = _ch
                                    )

                                    _rec_in_samples, _rec_in_sec = model.compute_receptive_field()
                                    
                                    # input 
                                    x = torch.randn(_b, _inp, _l+_rec_in_samples-1)
                                    c = torch.randn(_b, _n)
                                    h = torch.zeros(1, _b, 8)

                                    # 
                                    out = model(x, c, h)
                                    assert out.shape == expected_output.shape


                                    # concat gcn 
                                    model = FiLMTCN(
                                        n_inp = _inp,
                                        n_output = _outp,
                                        sample_rate = SR,
                                        n_cond = _n, 
                                        n_blocks = _nb,
                                        kernel_size = _k, 
                                        dilation_growth = _dl, 
                                        n_channels = _ch
                                    )

                                    _rec_in_samples, _rec_in_sec = model.compute_receptive_field()
                                    
                                    # input 
                                    x = torch.randn(_b, _inp, _l+_rec_in_samples-1)
                                    c = torch.randn(_b, _n)
                                    h = torch.zeros(1, _b, 8)

                                    # 
                                    out = model(x, c, h)
                                    assert out.shape == expected_output.shape


                                    # concat gcn 
                                    model = HyperTCN(
                                        n_inp = _inp,
                                        n_output = _outp,
                                        sample_rate = SR,
                                        n_cond = _n, 
                                        n_blocks = _nb,
                                        kernel_size = _k, 
                                        dilation_growth = _dl, 
                                        n_channels = _ch
                                    )

                                    _rec_in_samples, _rec_in_sec = model.compute_receptive_field()
                                    
                                    # input 
                                    x = torch.randn(_b, _inp, _l+_rec_in_samples-1)
                                    c = torch.randn(_b, _n)
                                    h = torch.zeros(1, _b, 8)

                                    # 
                                    out = model(x, c, h)
                                    assert out.shape == expected_output.shape




