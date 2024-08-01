import os 
import torch
import numpy as np 

from shutil import copyfile

import solver 
import utils
from dataset import SnapShot_AudioDataset

# ============================================================ #
# Config 
# ============================================================ #
# Load config from yaml files 
cmd = {
    'config': '/home/yytung/projects/pyneuralfx/frame_work/configs/rnn/lstm/snapshot_lstm.yml'
    #'config': '/home/yytung/projects/pyneuralfx/frame_work/configs/cnn/tcn/snapshot_tcn.yml'
}


args = utils.load_config(cmd['config'])
print(' > config:', cmd['config'])


# loss functions
loss_func_tra = utils.setup_loss_funcs(args) 
loss_func_val = utils.setup_loss_funcs(args) 
loss_funcs = [loss_func_tra, loss_func_val]


# device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(args.env.gpu_id)
args['device'] = device

# model 
model = utils.setup_models(args)

# expdir
LOAD_DIR = args.env.load_dir
print('EXP DIR: ', args.env.expdir)

PRE_ROOM = model.compute_receptive_field()[0] - 1
args['model']['pre_room'] = PRE_ROOM

# optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.train.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.train.lr_patience, verbose=True)


# to device
model.to(args.device)
for func in loss_funcs:
    func.to(args.device)

# ============================================================ #
# Functions
# ============================================================ #
def collate_fn(batch):
    wav_x_s = []
    wav_y_s = []
    #cond_s = []

    for idx in range(len(batch)):
        wav_x, wav_y = batch[idx]
        wav_x_s.append(wav_x[None, ...])
        wav_y_s.append(wav_y[None, ...])
        #cond_s.append([cond])

    x_final = np.concatenate(wav_x_s, axis=0)
    y_final = np.concatenate(wav_y_s, axis=0)
    #c_final = np.concatenate(cond_s, axis=0)

    return torch.from_numpy(x_final), torch.from_numpy(y_final), None #, torch.from_numpy(c_final)


def inference(path_savedir, exp_dir_val):
    global model
    print(' >>>>> inference')

    # load model 
    model = utils.load_model(
                exp_dir_val,
                model,
                device=args.device, 
                name='best_params.pt')
    
    # data
    valid_set = SnapShot_AudioDataset(
        input_data_path = args.data.test_x_path, 
        target_data_path = args.data.test_y_path, 
        sr=args.data.sampling_rate,
        win_len=None,
        pre_room=PRE_ROOM,)

    loader_valid = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.inference.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # validate
    path_outdir = os.path.join(exp_dir_val, path_savedir) 
    solver.validate(  
        args, 
        model, 
        loader_valid,
        loss_func_val, 
        path_save=path_outdir)
    
    amount, amount_train = model.compute_num_of_params()
    print(' > params amount: {:,d} | trainable: {:,d}'.format(amount, amount_train))

def train():
    global model

    if args.load_dir:
        print(' >>>>> fine-tuning')
        model = utils.load_model(
                LOAD_DIR,
                model,
                device=args.device, 
                name='best_params.pt')
    else:
        print(' >>>>> training')

    # datasets
    
    train_set = SnapShot_AudioDataset(
        input_data_path = args.data.train_x_path, 
        target_data_path = args.data.train_y_path, 
        sr=args.data.sampling_rate,
        win_len=args.data.buffer_size,
        pre_room=PRE_ROOM,)
    
    loader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print('> train dataset ready ...........')

    
    valid_set = SnapShot_AudioDataset(
        input_data_path = args.data.valid_x_path, 
        target_data_path = args.data.valid_y_path, 
        sr=args.data.sampling_rate,
        win_len=args.data.buffer_size,
        pre_room=PRE_ROOM,)
    
    
    loader_valid = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.train.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    

    print('> valid dataset ready ...........')
    os.makedirs(args['env']['expdir'], exist_ok=True)
    
    copyfile(__file__, os.path.join(args['env']['expdir'], os.path.basename(__file__)))
    copyfile(cmd['config'], os.path.join(args['env']['expdir'], os.path.basename(cmd['config'])))
    # training
    
    solver.train(
        args, 
        model, 
        loss_funcs, 
        optimizer,
        scheduler,
        loader_train, 
        valid_set=loader_valid,
        is_jit=args.env.is_jit)



# ============================================================ #
# Main  
# ============================================================ #
utils.check_configs(args)
train()
inference('valid_gen', args.env.expdir)

