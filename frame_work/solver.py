import os
import sys
import time
import datetime
import numpy as np
import soundfile as sf

import torch
from torch.nn.utils import clip_grad_norm_

import warnings
warnings.filterwarnings("ignore")


import utils
import saver


# ============================================================ #
# Train Non-Linear 
# ============================================================ #


HIDDEN_INTIAL = None
def _core(args, batch, model, h, loss_func):

    # data 
    wav_x, wav_y, vec_c = batch 
    wav_x = wav_x.float().to(args.device)
    wav_y = wav_y.float().to(args.device)
    if vec_c is not None:
        vec_c = vec_c.float().to(args.device)
    
    # initialization of hidden state 
    h = None
    if h is None and utils.FORWARD_TYPES[args.model.arch] != 5:
        # main network 
        h = torch.zeros(1, wav_x.shape[0], model.rnn_size).to(wav_x.device)
        cel = torch.zeros(1, wav_x.shape[0], model.rnn_size).to(wav_x.device)
        if utils.FORWARD_TYPES[args.model.arch] == 2 or utils.FORWARD_TYPES[args.model.arch] == 4:
            # hyper network 
            hyper_h = torch.zeros(1, wav_x.shape[0], model.hyper_rnn_size).to(wav_x.device)
            hyper_cel = torch.zeros(1, wav_x.shape[0], model.hyper_rnn_size).to(wav_x.device)
        
    if utils.FORWARD_TYPES[args.model.arch] == 1:
        wav_y_pred, h, _ = model(wav_x, vec_c, h)
    elif utils.FORWARD_TYPES[args.model.arch] == 2:
        wav_y_pred, h, _ = model(wav_x, vec_c, h, hyper_h)
    elif utils.FORWARD_TYPES[args.model.arch] == 3:
        wav_y_pred, h, _ = model(wav_x, vec_c, (h, cel))
    elif utils.FORWARD_TYPES[args.model.arch] == 4:
        wav_y_pred, h, _ = model(wav_x, vec_c, (h, cel), (hyper_h, hyper_cel))
    elif utils.FORWARD_TYPES[args.model.arch] == 5:
        wav_y_pred = model(wav_x, vec_c)
    

    # loss
    if loss_func:
        loss = loss_func(wav_y_pred, wav_y)
        return loss, (wav_y, wav_y_pred), h
    else:
        return wav_y_pred, h

def validate(
        args, 
        model, 
        data_set,
        loss_func, 
        path_save=None,
        write_sr=None):
    
    # eval mode
    model.eval()
    # init
    results_pred = []
    results_anno = []
    results_inp  = []
    fn_list = []

    rtf_all = []
    
    with torch.no_grad():
        list_loss = []
        num_batch = len(data_set) 
        
        # run validation
        h = HIDDEN_INTIAL
        for bidx, batch in enumerate(data_set):
            sys.stdout.write('{}/{} \r'.format(bidx, num_batch))
            sys.stdout.flush()  

            # get batch 
            wav_x, wav_y, c = batch

            # for naming only 
            if c is not None:
                cond = utils.convert_tensor_to_numpy(c, is_squeeze=True)
                condfn = []
                for idx, subc in enumerate(cond):
                    idx = idx % args.data.num_conds
                    __max_cond = float(max(args.data.norm_tensor[idx]))
                    __min_cond = float(min(args.data.norm_tensor[idx]))
                    
                    _tmp = ((subc + 1) / 2) * __max_cond + __min_cond
                    condfn.append(_tmp)
            else:
                condfn = []
                        
            # run model
            st_time = time.time()

            loss, (wav_y, wav_y_pred), h = _core(args, batch, model, h, loss_func)
            
            ed_time = time.time()

            # RTF
            run_time = ed_time - st_time
            song_time = wav_y.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            rtf_all.append(rtf)

            wav_x = wav_x.reshape(-1).unsqueeze(0)
            wav_y = wav_y.reshape(-1).unsqueeze(0)
            wav_y_pred = wav_y_pred.reshape(-1).unsqueeze(0)
            # append result
            if path_save:
                results_pred.append(
                    utils.convert_tensor_to_numpy(wav_y_pred, is_squeeze=True)
                )
                results_anno.append(
                    utils.convert_tensor_to_numpy(wav_y, is_squeeze=True)
                )
                results_inp.append(
                    utils.convert_tensor_to_numpy(wav_x, is_squeeze=True)[args.model.pre_room:]
                )

                new_fn_name = f'output{bidx}'
                if len(condfn) != 0:
                    for i in range(args.data.num_conds):
                        new_fn_name += f'_{condfn[i]:.1f}'
                new_fn_name = new_fn_name + '.wav'
                
                fn_list.append(
                    new_fn_name
                )
            # append loss
            list_loss.append(loss.item())
            
    if path_save:
        print('mean loss:', np.mean(list_loss))
        os.makedirs(path_save, exist_ok=True) 

        # save pred
        path_outdir_pred = os.path.join(path_save, 'pred')
        path_outdir_anno = os.path.join(path_save, 'anno')
        path_outdir_inp  = os.path.join(path_save, 'inp')

        os.makedirs(path_outdir_pred, exist_ok=True)
        os.makedirs(path_outdir_anno, exist_ok=True)
        os.makedirs(path_outdir_inp,  exist_ok=True)

        for idx, fn in enumerate(fn_list):
            print('---------------------------------')
            print(' >>>', idx, fn)
            
            path_outfile_pred = os.path.join(path_outdir_pred, fn)
            path_outfile_anno = os.path.join(path_outdir_anno, fn)
            path_outfile_inp = os.path.join(path_outdir_inp,  fn)

            print(' > path_outfile pred:', path_outfile_pred)
            print(' > path_outfile anno:', path_outfile_anno)
            print(' > path_outfile inp:' , path_outfile_inp)

            sample_pred = results_pred[idx]
            sample_anno = results_anno[idx]
            sample_inp  = results_inp[idx]

            if args.data.out_channels > 1:
                sample_pred = np.transpose(sample_pred, (1, 0))
                sample_anno = np.transpose(sample_anno, (1, 0))
                sample_inp  = np.transpose(sample_inp,  (1, 0))

            print(' > sample shape:', sample_pred.shape)
            print(' > sample shape:', sample_anno.shape)
            print(' > sample shape:', sample_inp.shape)
            
            if write_sr is None:
                write_sr = args.data.sampling_rate 
            
            sf.write(path_outfile_pred, sample_pred, write_sr, subtype='PCM_24')
            sf.write(path_outfile_anno, sample_anno, write_sr, subtype='PCM_24')
            sf.write(path_outfile_inp,  sample_inp,  write_sr, subtype='PCM_24')
        
    return np.mean(list_loss)

def train(
    args, 
    model, 
    loss_funcs, 
    optimizer,
    scheduler,
    data_set, 
    valid_set=None,
    is_jit=False
):
    
    # create saver
    print(' [!] saver created!')
    saver_agent = saver.Saver(args.env.expdir, debug=args.env.debug_mode)

    # jit model
    if is_jit:
        model = torch.jit.script(model)
    
    # compute model size
    batch_size = args.train.batch_size 
    amount, amount_train = model.compute_num_of_params()
    log_amount = ' > params amount: {:,d} | trainable: {:,d} |  bs: {:,d}  '.format(amount, amount_train, batch_size)
    saver_agent.log_info(log_amount)

    # training config
    model.train()
    is_valid = True if valid_set is not None else False
    num_batch = len(data_set) 
    best_loss = 9999
    max_grad_norm = 3   
    
    # timer
    print('{:=^40}'.format(' start training '))
    time_start_train = time.time()
    acc_batch_time = 0

    # start training
    valid_counter = 0
    for epoch in range(args.train.epochs):
        h = HIDDEN_INTIAL
        for bidx, batch in enumerate(data_set):
            time_start_batch = time.time()
            
            # counters
            saver_agent.global_step_increment()
            
            # run model
            loss, (wav_y, wav_y_pred), h = _core(args, batch, model, h, loss_funcs[0])
            
            # update
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm) # rnn
            optimizer.step()
        
            
            # monitoring
            ## print loss
            if saver_agent.global_step % args.train.interval_log == 0: 
                # time
                acc_batch_time += time.time() - time_start_batch
                train_time = time.time() - time_start_train
                
                # check progress
                log = 'epoch: %d/%d (%3d/%3d) | %s | t: %.2f | loss: %.6f | time: %s | counter: %d' % (
                    epoch, 
                    args.train.epochs, 
                    bidx, 
                    num_batch, 
                    args.env.expdir,
                    acc_batch_time,
                    loss.item(), 
                    str(datetime.timedelta(seconds=train_time))[:-5],
                    saver_agent.global_step)
                saver_agent.log_info(log)

                # check value range
                log_range = "pred: max:{:.6f}, min:{:.6f}, mean:{:.6f}\n" \
                            "anno: max:{:.6f}, min:{:.6f}, mean:{:.6f}".format(
                                torch.max(wav_y_pred), torch.min(wav_y_pred), torch.mean(wav_y_pred), 
                                torch.max(wav_y),      torch.min(wav_y), torch.mean(wav_y))
                
                # write/save log
                saver_agent.log_info(log_range)
                saver_agent.log_loss({'train loss': loss.item()})

                # re-calculate time
                acc_batch_time = 0

            ## validation
            if saver_agent.global_step % args.train.interval_val == 0 and is_valid:
                print(' [*] run validation...')

                # compute loss
                
                loss_valid = validate(
                    args, 
                    model, 
                    valid_set,
                    loss_funcs[1]
                )
                model.train()
                
                # log
                log = ' > validation loss: %.6f | counter: %d' % (
                    loss_valid, saver_agent.global_step)
                saver_agent.log_info(log)

                # save/write log
                saver_agent.log_loss({'valid loss': loss_valid})
                valid_counter += 1 
                # save best validation model
                if loss_valid < best_loss:
                    valid_counter = 0  
                    saver_agent.log_info(' [!] --- best model updated ---')
                    saver_agent.save_model(model, outdir=args.env.expdir, name='best')
                    best_loss = loss_valid
                
                scheduler.step(loss_valid)

                # tolerance
                if valid_counter >= args.train.improvement_patience:
                    return 
                
            ## save model
            if saver_agent.global_step % args.train.interval_ckpt == 0:
                # save
                saver_agent.save_model(
                    model, outdir=args.env.expdir, name='latest')

                # make loss report
                saver_agent.make_loss_report()


