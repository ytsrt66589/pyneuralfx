import os 


import librosa 
import torch 
import numpy as np
import json 
from tqdm.contrib import tzip
from tqdm import tqdm
from utils import traverse_dir
from pyneuralfx.loss_func.loss_func import * 
from pyneuralfx.eval_metrics.eval_metrics import * 

SR = 44100


REPORTED_METRICS = {
    'MRSTFTLoss': MRSTFTLoss(
        scales=[2048]
    ),
    'ESRLoss': ESRLoss(),
    'Transient': TransientPreservation_v2(SR),
    'LUFS': LUFS(sr=SR),
    'CrestFactor': CrestFactor(),
    'RMSEnergy': RMSEnergy(),
    'SpectralCentroid': SpectralCentroid(),

}


exp_names = [
    'snapshot_example', 
]


for exp_name in exp_names:
    print('> exp name: ', exp_name)

    for t in ['valid_gen']:

        path_json = os.path.join('exp',  exp_name, t)

        path_gt = os.path.join('exp',  exp_name, t, 'anno')
        path_pred = os.path.join('exp',  exp_name, t, 'pred')


        filelist_gt = traverse_dir(path_gt, is_pure=True, is_sort=True)
        filelist_pred = traverse_dir(path_pred, is_pure=True, is_sort=True)
        assert len(filelist_gt) == len(filelist_pred)
        print('> total audio clips: ', len(filelist_pred))

        loss_record = {}
        loss_record.clear()

        # initial loss record 
        for k in REPORTED_METRICS:
            loss_record[k] = []
        
        for (gt_fn, pred_fn) in tzip(filelist_gt, filelist_pred):
            
            wav_gt, sr_gt = librosa.load(os.path.join(path_gt, gt_fn), sr=None, mono=True)
            wav_pred, sr_pred = librosa.load(os.path.join(path_pred, pred_fn), sr=None, mono=True)
            
            assert gt_fn == pred_fn
            assert sr_gt == sr_pred
            assert sr_gt == SR 

            # to torch 
            wav_gt = torch.from_numpy(wav_gt).unsqueeze(0).unsqueeze(0).float()
            wav_pred = torch.from_numpy(wav_pred).unsqueeze(0).unsqueeze(0).float()
            
            for k in REPORTED_METRICS:
                metri = REPORTED_METRICS[k]
                score = metri(wav_pred, wav_gt)
                loss_record[k].append(score.item())
            
        
        print( '##################################################')
        print(f'#####    evaluation report   {t, exp_name}  #########')
        print( '##################################################')
        final_reports = {}
        final_reports.clear()
        for r in loss_record:
            loss_value = np.mean(loss_record[r])
            print(f'> metric {r}: ', loss_value)
            final_reports[r] = loss_value
        print('#########################################')
    

    file_names = os.path.basename(exp_name)
    
    with open(os.path.join(path_json, f'{file_names}_metric.json'), 'w') as f:
        json.dump(final_reports, f)
    final_reports.clear()