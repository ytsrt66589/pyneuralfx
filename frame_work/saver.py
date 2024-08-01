import os
import json
import time
import torch
import logging
import datetime
import collections
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


from json import JSONEncoder
class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(json.NpEncoder, self).default(obj)

# ============================================================ #
# Utilities
# ============================================================ #

def to_json(path_params, path_json, cnn_mask_bounds=None):
    params = torch.load(path_params, map_location=torch.device('cpu'))
    raw_state_dict = {}
    for k, v in params.items():
        # print('-----------')
        # print(k)
        # print(v.shape)
        val = v.flatten().numpy().tolist()
        if ('conv_in' in k) and (cnn_mask_bounds is not None):
            val = val[cnn_mask_bounds[0]:cnn_mask_bounds[1]]
        raw_state_dict[k] = val

    with open(path_json, 'w') as outfile:
        json.dump(raw_state_dict, outfile,indent= "\t")

    

def make_loss_report(
        path_log,
        path_figure='loss.png',
        dpi=100):

    # load logfile
    monitor_vals = collections.defaultdict(list)
    with open(path_log, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                key, val, step, acc_time = line.split(' | ')
                monitor_vals[key].append((float(val), int(step), acc_time))
            except:
                continue

    # collect
    step_train = [item[1] for item in monitor_vals['train loss']]
    vals_train = [item[0] for item in monitor_vals['train loss']]

    step_valid = [item[1] for item in monitor_vals['valid loss']]
    vals_valid = [item[0] for item in monitor_vals['valid loss']]


    x_min = step_valid[np.argmin(vals_valid)]
    y_min = min(vals_valid)

    # plot
    fig = plt.figure(dpi=dpi)
    plt.title('training process')
    plt.plot(step_train, vals_train, label='train')
    plt.plot(step_valid, vals_valid, label='valid')
    plt.yscale('log')
    plt.plot([x_min], [y_min], 'ro')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_figure)


# ============================================================ #
# Saver Object
# ============================================================ #


class Saver(object):
    def __init__(
            self, 
            exp_dir,
            debug=False):

        # exp dir
        self.exp_dir = exp_dir
        
        # cold start
        self.global_step = -1
        self.init_time = time.time()

        # makedirs
        os.makedirs(exp_dir, exist_ok=debug)

        # path
        self.path_log_loss = os.path.join(exp_dir, 'log_loss.txt')
        self.path_log_info = os.path.join(exp_dir, 'log_info.txt')

    def log_info(self, msg_str):
        print(msg_str)
        with open(self.path_log_info, 'a') as fp:
            fp.write(msg_str+'\n')

    def log_loss(self, loss_dict):
        cur_time = time.time() - self.init_time
        step = self.global_step

        with open(self.path_log_loss, 'a') as fp:
            for key, val in loss_dict.items():
                msg_str = '{:10s} | {:.10f} | {:10d} | {}\n'.format(
                    key, 
                    val, 
                    step, 
                    cur_time
                )
                fp.write(msg_str)

    def save_model(
            self, 
            model, 
            outdir=None, 
            name='model',
            is_to_json=True):

        # path
        if outdir is None:
            outdir = os.path.join('ckpts', self.exp_dir)
        path_pt = os.path.join(outdir, name+'.pt')
        path_params = os.path.join(outdir, name+'_params.pt')
       
        # check
        print(' [*] saving model to {}, name: {}'.format(outdir, name))

        # save
        # torch.save(model, path_pt)
        torch.save(model.state_dict(), path_params)

        # dump to json
        if is_to_json:
            path_json = os.path.join(outdir, name+'_params.json')
            to_json(path_params, path_json)
    
        path_plugin_json = os.path.join(outdir, name+'_rt_plugin.json')
        with open(path_plugin_json, 'w') as json_file:
            json.dump(model.state_dict(), json_file, cls=EncodeTensor)

    def global_step_increment(self):
        self.global_step += 1

    def make_loss_report(self):
        path_figure = os.path.join(self.exp_dir, 'loss_report.png')
        make_loss_report(
            self.path_log_loss,
            path_figure=path_figure,
            dpi=100)

