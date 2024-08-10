import torch 
import utils
#from pyneuralfx.vis.plotting import * 
from pyneuralfx.vis.sysplotting import * 
from pyneuralfx.models.rnn.gru import ConcatGRU


path_outdir = '.'
num_channel = 1

nn_model = None
num_conds = 2 
sr = 48000
gain = 1
freq = 100

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

def nn_forwading(x, nn_model, model_arch, num_conds):
    
    wav_x = x.float().to('cpu')
    model = nn_model.to('cpu')

    vec_c = None 
    if num_conds > 0:
        vec_c = torch.zeros(1, num_conds).float().to('cpu')

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

cmd = {
    #'config': '/home/yytung/projects/pyneuralfx/frame_work/exp/boss_od3/statichyper_gru_32/statichyper_gru.yml'
    'config': '/home/yytung/projects/pyneuralfx/frame_work/exp/boss_od3/film_gru_32/film_gru.yml'
}

args = utils.load_config(cmd['config'])

nn_model = utils.setup_models(args)
nn_model = utils.load_model(
                args.env.expdir,
                nn_model,
                device='cpu', 
                name='best_params.pt')


# system plotting 
#plot_distortion_curve(path_outdir, args.data.sampling_rate, gain, freq, nn_forwading, nn_model, args.model.arch, args.data.num_conds)
plot_harmonic_response(path_outdir, args.data.sampling_rate, gain, freq, nn_forwading, nn_model, args.model.arch, args.data.num_conds)
plot_sine_sweep_response_spec(path_outdir, args.data.sampling_rate, nn_forwading, nn_model, args.model.arch, args.data.num_conds)

# wav comparison 

#path_pred = '/home/yytung/projects/pyneuralfx/frame_work/exp/boss_od3/concat_gru_32/valid_gen/pred/output10_3.0_4.0.wav'
#path_anno = '/home/yytung/projects/pyneuralfx/frame_work/exp/boss_od3/concat_gru_32/valid_gen/anno/output10_3.0_4.0.wav'
#plot_spec_diff(path_anno, path_pred, '.', args.data.sampling_rate, 1)