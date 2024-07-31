import torch 
from pyneuralfx import pyneuralfx


from pyneuralfx.models.rnn.gru import *

B = [1, 32]
INP = [1, 2]
OUTP = [1, 2]
N_CONDS = [1, 2]
LENGTH = [8, 32] 
NUM_BLOCKS = [1, 2]
SR = 48000
RNN_SIZE = [1, 4, 8] 
HYPER_RNN_SIZE = [4]
KERNEL_SIZES = [3, 5]
DILATION_GROWTHS = [2]
NCHANNELS = [8]

for _b in B:
    for _inp in INP:
        for _outp in OUTP:
            for _n in N_CONDS:
                for _nn_size in RNN_SIZE:
                    for _l in LENGTH:
                        for _ln in [True, False]:
                            # output 
                            expected_output = torch.randn(_b, _outp, _l)
                                                    
                            # snapshot 
                            model = SnapShotGRU(
                                inp_channel =  _inp,
                                out_channel = _outp,
                                rnn_size = _nn_size,
                                sample_rate = SR, 
                                layer_norm = _ln
                            )
                                            
                            _rec_in_samples, _rec_in_sec = model.compute_receptive_field()
                                                    
                            # input 
                            x = torch.randn(_b, _inp, _l+_rec_in_samples-1)
                            c = torch.randn(_b, _n)
                            h = torch.zeros(1, _b, _nn_size)

                            # 
                            out, _, _ = model(x, c, h)
                            assert out.shape == expected_output.shape


                            model = ConcatGRU(
                                inp_channel =  _inp,
                                out_channel = _outp,
                                rnn_size = _nn_size,
                                sample_rate = SR, 
                                num_conds = _n,
                                layer_norm = _ln
                            )
                                            
                            _rec_in_samples, _rec_in_sec = model.compute_receptive_field()
                                                    
                            # input 
                            x = torch.randn(_b, _inp, _l+_rec_in_samples-1)
                            c = torch.randn(_b, _n)
                            h = torch.zeros(1, _b, _nn_size)

                            # 
                            out, _, _ = model(x, c, h)
                            assert out.shape == expected_output.shape


                            model = FiLMGRU(
                                inp_channel =  _inp,
                                out_channel = _outp,
                                rnn_size = _nn_size,
                                sample_rate = SR, 
                                num_conds = _n,
                                layer_norm = _ln
                            )
                                            
                            _rec_in_samples, _rec_in_sec = model.compute_receptive_field()
                                                    
                            # input 
                            x = torch.randn(_b, _inp, _l+_rec_in_samples-1)
                            c = torch.randn(_b, _n)
                            h = torch.zeros(1, _b, _nn_size)

                            # 
                            out, _, _ = model(x, c, h)
                            assert out.shape == expected_output.shape


                            model = StaticHyperGRU(
                                inp_channel =  _inp,
                                out_channel = _outp,
                                rnn_size = _nn_size,
                                sample_rate = SR, 
                                num_conds = _n,
                                layer_norm = _ln
                            )
                                            
                            _rec_in_samples, _rec_in_sec = model.compute_receptive_field()
                                                    
                            # input 
                            x = torch.randn(_b, _inp, _l+_rec_in_samples-1)
                            c = torch.randn(_b, _n)
                            h = torch.zeros(1, _b, _nn_size)

                            # 
                            out, _, _ = model(x, c, h)
                            assert out.shape == expected_output.shape
                            

                            model = DynamicHyperGRU(
                                inp_channel =  _inp,
                                out_channel = _outp,
                                rnn_size = _nn_size,
                                hyper_rnn_size = _nn_size,
                                sample_rate = SR, 
                                num_conds = _n,
                                layer_norm = _ln
                            )
                                            
                            _rec_in_samples, _rec_in_sec = model.compute_receptive_field()
                                                    
                            # input 
                            x = torch.randn(_b, _inp, _l+_rec_in_samples-1)
                            c = torch.randn(_b, _n)
                            h = torch.zeros(1, _b, _nn_size)
                            h_hat = torch.zeros(1, _b, _nn_size)
                            
                            # 
                            out, _, _ = model(x, c, h, h_hat)
                            assert out.shape == expected_output.shape





