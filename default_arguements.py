ml_args = {
    'batch_size':64,
    'n_epochs':1,
    'n_steps':None,
    'warmup':0.0,
    'lr_decay':'steplr',
    'lr_decay_stepsize': 1,
    'lr_decay_gamma': 0.1,
    'lr_decay_step_timing': 'epoch',
    'n_saves':2,
    'validation_criteria':'min',
    'optimizer':'adadelta',
    'lr':1.0,
    'grad_clip':None,
    'loss_type':'NLL',
    'affine_aug':False,
    'add_Gnoise':False,
    'gaussian_std':1.0,
    'normalization_percentiles':'auto_min_max',
    'normalization_channels':'auto',
    'n_workers':0,
    'visualize_val':False,
    'data_parallel':False
}

dt_args={
    'silo_dtype': 'np.uint8',
    'numpy_shape': (28,28),
    'pad_shape':None,
    'pre_load':True,
    'one_hot_encode': False,
    'max_classes': 10
}

mdl_args = {
    'kernel': 5,
    'hidden_layer_parameters': 128,
    'output_style': 'softmax',
    'input_shape': (28, 28)
}