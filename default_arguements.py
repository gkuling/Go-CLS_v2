ml_args = {
    'batch_size':64,
    'n_epochs':1,
    'n_steps':None,
    'warmup':0.0,
    'lr_decay':None,
    'n_saves':2,
    'validation_criteria':'min',
    'optimizer':'adadelta',
    'lr':1.0,
    'grad_clip':None,
    'loss_type':'MSE',
    'affine_aug':False,
    'add_Gnoise':False,
    'gaussian_std':1.0,
    'normalization_percentiles':'auto_min_max',
    'normalization_channels':[(33.0323807385913, 3*76.96857351487527)],
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