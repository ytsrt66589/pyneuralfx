import os
import json
import pickle
import argparse
import numpy as np
import soundfile as sf

import scipy
import scipy.signal
import yaml 
import torch

from pyneuralfx.loss_func.loss_func import * 

from pyneuralfx.models.rnn.gru import * 
from pyneuralfx.models.rnn.lstm import * 
from pyneuralfx.models.rnn.rnn import * 

from pyneuralfx.models.cnn.tcn import * 
from pyneuralfx.models.cnn.gcn import * 


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

     
def load_model(
        path_exp, 
        model,
        device='cpu', 
        name='model_params.pt'):

    # check
    path_pt = os.path.join(path_exp, name)
    print(' [*] restoring model from', path_pt)

    model.load_state_dict(torch.load(path_pt, map_location=torch.device(device)))
    return model


def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    # print(args)
    return args


def traverse_dir(
        root_dir,
        extension='.wav',
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def setup_models(args):

    # > ====================================== <
    # rnn 
    #    snapshot-gru, snapshot-lstm, snapshot-valilla-rnn
    #    concat-gru, concat-lstm, concat-vanilla-rnn 
    #    film-gru,   film-lstm,   film-vanilla-rnn 
    #    statichyper-gru, statichyper-lstm, statichyper-vanilla-rnn
    #    dynamichyper-gru, dynamichyper-lstm
    # cnn 
    #    snapshot-gcn, snapshot-tcn 
    #    concat-gcn, concat-tcn 
    #    film-gcn,   film-tcn 
    #    hyper-gcn,  hyper-tcn 
    # > ====================================== <

    model = None 

    if args.model.arch == 'snapshot-gru':
        model = SnapShotGRU(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'concat-gru':
        model = ConcatGRU(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'film-gru':
        model = FiLMGRU(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            n_mlp_blocks = args.model.n_mlp_blocks,
            mlp_size = args.model.mlp_size,
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'statichyper-gru':
        model = StaticHyperGRU(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            n_mlp_blocks = args.model.n_mlp_blocks,
            mlp_size = args.model.mlp_size,
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'dynamichyper-gru':
        model = DynamicHyperGRU(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            hyper_rnn_size= args.model.hyper_rnn_size,
            n_z_size = args.model.n_z_size,
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'snapshot-lstm':
        model = SnapShotLSTM(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'concat-lstm':
        model = ConcatLSTM(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'film-lstm':
        model = FiLMLSTM(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            n_mlp_blocks = args.model.n_mlp_blocks,
            mlp_size = args.model.mlp_size,
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'statichyper-lstm':
        model = StaticHyperLSTM(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            n_mlp_blocks = args.model.n_mlp_blocks,
            mlp_size = args.model.mlp_size,
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'dynamichyper-lstm':
        model = DynamicHyperLSTM(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            hyper_rnn_size= args.model.hyper_rnn_size,
            n_z_size = args.model.n_z_size,
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'snapshot-vanilla-rnn':
        model = SnapShotVanillaRNN(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'concat-vanilla-rnn':
        model = ConcatVanillaRNN(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'film-vanilla-rnn':
        model = FiLMVanillaRNN(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            n_mlp_blocks = args.model.n_mlp_blocks,
            mlp_size = args.model.mlp_size,
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'statichyper-vanilla-rnn':
        model = StaticHyperVanillaRNN(
            inp_channel = args.data.inp_channels,
            out_channel = args.data.out_channels,
            rnn_size = args.model.main_rnn_hidden_size,
            sample_rate = args.data.sampling_rate, 
            n_mlp_blocks = args.model.n_mlp_blocks,
            mlp_size = args.model.mlp_size,
            num_conds = args.data.num_conds,
            layer_norm = args.model.layer_norm,
            rnn_bias = args.model.rnn_bias, 
        )
    elif args.model.arch == 'snapshot-gcn':
        model = SnapShotGCN(
            n_inp = args.data.inp_channels, 
            n_output = args.data.out_channels, 
            sample_rate = args.data.sampling_rate,
            n_blocks = args.model.n_blocks, 
            kernel_size = args.model.kernel_size, 
            dilation_growth = args.model.dilation_growth, 
            n_channels = args.model.n_channels, 
            causal = args.model.causal, 
        )
    elif args.model.arch == 'concat-gcn':
        model = ConcatGCN(
            n_inp = args.data.inp_channels, 
            n_output = args.data.out_channels, 
            n_cond = args.data.num_conds,
            sample_rate = args.data.sampling_rate,
            n_blocks = args.model.n_blocks, 
            kernel_size = args.model.kernel_size, 
            dilation_growth = args.model.dilation_growth, 
            n_channels = args.model.n_channels, 
            causal = args.model.causal, 
        )
    elif args.model.arch == 'film-gcn':
        model = FiLMGCN(
            n_inp = args.data.inp_channels, 
            n_output = args.data.out_channels, 
            n_cond = args.data.num_conds,
            sample_rate = args.data.sampling_rate,
            n_blocks = args.model.n_blocks, 
            kernel_size = args.model.kernel_size, 
            dilation_growth = args.model.dilation_growth, 
            n_channels = args.model.n_channels, 
            causal = args.model.causal, 
            pre_film_size = args.model.pre_film_size,
            pre_film_blocks = args.model.pre_film_blocks,
        )
    elif args.model.arch == 'hyper-gcn':
        model = HyperGCN(
            n_inp = args.data.inp_channels, 
            n_output = args.data.out_channels, 
            n_cond = args.data.num_conds,
            sample_rate = args.data.sampling_rate,
            n_blocks = args.model.n_blocks, 
            kernel_size = args.model.kernel_size, 
            dilation_growth = args.model.dilation_growth, 
            n_channels = args.model.n_channels, 
            causal = args.model.causal, 
        )
    elif args.model.arch == 'snapshot-tcn':
        model = SnapShotTCN(
            n_inp = args.data.inp_channels, 
            n_output = args.data.out_channels, 
            sample_rate = args.data.sampling_rate,
            n_blocks = args.model.n_blocks, 
            kernel_size = args.model.kernel_size, 
            dilation_growth = args.model.dilation_growth, 
            n_channels = args.model.n_channels, 
            causal = args.model.causal, 
        )
    elif args.model.arch == 'concat-tcn':
        model = ConcatTCN(
            n_inp = args.data.inp_channels, 
            n_output = args.data.out_channels, 
            n_cond = args.data.num_conds,
            sample_rate = args.data.sampling_rate,
            n_blocks = args.model.n_blocks, 
            kernel_size = args.model.kernel_size, 
            dilation_growth = args.model.dilation_growth, 
            n_channels = args.model.n_channels, 
            causal = args.model.causal, 
        )
    elif args.model.arch == 'film-tcn':
        model = FiLMTCN(
            n_inp = args.data.inp_channels, 
            n_output = args.data.out_channels, 
            n_cond = args.data.num_conds,
            sample_rate = args.data.sampling_rate,
            n_blocks = args.model.n_blocks, 
            kernel_size = args.model.kernel_size, 
            dilation_growth = args.model.dilation_growth, 
            n_channels = args.model.n_channels, 
            causal = args.model.causal, 
            pre_film_size = args.model.pre_film_size,
            pre_film_blocks = args.model.pre_film_blocks,
        )
    elif args.model.arch == 'hyper-tcn':
        model = HyperTCN(
            n_inp = args.data.inp_channels, 
            n_output = args.data.out_channels, 
            n_cond = args.data.num_conds,
            sample_rate = args.data.sampling_rate,
            n_blocks = args.model.n_blocks, 
            kernel_size = args.model.kernel_size, 
            dilation_growth = args.model.dilation_growth, 
            n_channels = args.model.n_channels, 
            causal = args.model.causal, 
        )

    return model


def setup_loss_funcs(args, customized_loss_func=None):
    if customized_loss_func is not None:
        return customized_loss_func

    if args.loss.loss_func == 'esr_loss':
        print('> ========= ESR ============== <')
        return ESRLoss(pre_emp=args.loss.pre_emp)
    elif args.loss.loss_func == 'hybrid_loss':
        print('> ========= Hybrid ============== <')
        return HybridLoss(pre_emp=args.loss.pre_emp)
    elif args.loss.loss_func == 'complex_stft_loss':
        print('> ========= Complex STFT ============== <')
        return STFTLoss()
    return None


def check_configs(args):
    if args.data.num_conds:
        assert len(args.data.norm_tensor) == args.data.num_conds
    

FORWARD_TYPES = {
    'snapshot-gru': 1, 
    'concat-gru': 1,
    'film-gru': 1,
    'statichyper-gru': 1,
    'dynamichyper-gru': 2, 
    'snapshot-lstm': 3,
    'concat-lstm': 3,
    'film-lstm': 3,
    'statichyper-lstm': 3,
    'dynamichyper-lstm': 4,
    'snapshot-vanilla-rnn': 1,
    'film-vanilla-rnn': 1,
    'statichyper-vanilla-rnn': 1,

    'snapshot-gcn': 5,
    'concat-gcn': 5,
    'film-gcn': 5,
    'hyper-gcn': 5,
    'snapshot-tcn': 5,
    'concat-tcn': 5,
    'film-tcn': 5,
    'hyper-tcn': 5,

}


def forward_func(x, cond, nn_model, model_arch, device):
    
    wav_x = x.float().to(device)
    model = nn_model.to(device)

    vec_c = None 
    vec_c = torch.from_numpy(
        np.array([cond])
    ).float().to(device)
    
    # initialization of hidden state 
    h = None
    if h is None and FORWARD_TYPES[model_arch] != 5:
        # main network 
        h = torch.zeros(1, wav_x.shape[0], model.rnn_size).to(wav_x.device)
        cel = torch.zeros(1, wav_x.shape[0], model.rnn_size).to(wav_x.device)
        if FORWARD_TYPES[model_arch] == 2 or FORWARD_TYPES[model_arch] == 4:
            # hyper network 
            hyper_h = torch.zeros(1, wav_x.shape[0], model.hyper_rnn_size).to(wav_x.device)
            hyper_cel = torch.zeros(1, wav_x.shape[0], model.hyper_rnn_size).to(wav_x.device)
    
    if FORWARD_TYPES[model_arch] == 1:
        wav_y_pred, h, _ = model(wav_x, vec_c, h)
    elif FORWARD_TYPES[model_arch] == 2:
        wav_y_pred, h, _ = model(wav_x, vec_c, h, hyper_h)
    elif FORWARD_TYPES[model_arch] == 3:
        wav_y_pred, h, _ = model(wav_x, vec_c, (h, cel))
    elif FORWARD_TYPES[model_arch] == 4:
        wav_y_pred, h, _ = model(wav_x, vec_c, (h, cel), (hyper_h, hyper_cel))
    elif FORWARD_TYPES[model_arch] == 5:
        wav_y_pred = model(wav_x, vec_c)

    return wav_y_pred