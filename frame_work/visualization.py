import torch 
import utils
from pyneuralfx.vis.plotting import * 
from pyneuralfx.vis.sysplotting import * 


path_outdir = '.'
num_channel = 1

nn_model = None
num_conds = 2 
sr = 48000
gain = 1
freq = 100

cmd = {
    'config': '/home/yytung/projects/pyneuralfx/frame_work/exp/boss_od3/film_gru_32/film_gru.yml'
}

args = utils.load_config(cmd['config'])
nn_model = utils.setup_models(args)
nn_model = utils.load_model(
                args.env.expdir,
                nn_model,
                device='cpu', 
                name='best_params.pt')

# device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# system plotting 
#plot_harmonic_response(path_outdir, args.data.sampling_rate, gain, freq, utils.forward_func, [1, 1], nn_model,  args.model.arch, device)
plot_sine_sweep_response_spec(path_outdir, args.data.sampling_rate, utils.forward_func, [1, 1],nn_model,  args.model.arch,  device)

# wav comparison 
#path_pred = '/home/yytung/projects/pyneuralfx/frame_work/exp/boss_od3/concat_gru_32/valid_gen/pred/output10_3.0_4.0.wav'
#path_anno = '/home/yytung/projects/pyneuralfx/frame_work/exp/boss_od3/concat_gru_32/valid_gen/anno/output10_3.0_4.0.wav'
#plot_spec_diff(path_anno, path_pred, '.', args.data.sampling_rate, 1)