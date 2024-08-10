# Configurations 
This document describe some parameters in the configuration file. Guiding users to modify it efficiently. 

We provide one example of detail configurations for all supported models in `configs/`. 


Take one example: 

## Snapshot modeling 
```
data:
  buffer_size: 16384 # number of input audio samples for training (if cnn-based, then will add additional receptive field - 1)
  sampling_rate: 44100 
  inp_channels: 1 
  out_channels: 1 
  train_x_path: /path/to/train_x 
  train_y_path: /path/to/train_y
  valid_x_path: /path/to/valid_x
  valid_y_path: /path/to/valid_y
  test_x_path: /path/to/test_x
  test_y_path: /path/to/test_y
model:
  arch: snapshot-gcn # architecture for training
  n_blocks: 9
  kernel_size: 3
  dilation_growth: 2
  n_channels: 16
  causal: True 
loss:
  loss_func: 'hybrid_loss' 
  pre_emp: True 
device: cuda
env:
  expdir: /path/to/exp_record
  gpu: 0
  gpu_id: 0
  is_jit: False
  debug_mode: True
inference:
  batch_size: 1
train:
  batch_size: 25
  epochs: 500
  interval_log: 10
  interval_ckpt: 450
  interval_val: 450
  lr_patience: 2
  improvement_patience: 6
  lr: 0.001
```

---
For model.arch, PyNeuralFx supports following options:

**Snapshot modeling**
```
snapshot-vanilla-rnn
snapshot-gru
snapshot-lstm
snapshot-gcn
snapshot-tcn
```


**Full modeling**
```
concat-gru
film-gru
statichyper-gru
dynamichyper-gru
concat-lstm
film-lstm
statichyper-lstm
dynamichyper-lstm
film-vanilla-rnn
statichyper-vanilla-rnn
concat-gcn
film-gcn
hyper-gcn
concat-tcn
film-tcn
hyper-tcn
```

---

For loss.loss_func, PyNeuralFx supports following options:

```
hybrid_loss: (MAE + MRSTFT)
esr_loss: (ESR) 
stft_complex_loss: (STFT Complex Loss)
customized: (You can customized your training loss based on the loss function we provided) 
```

---