'''
Copyright (c) 2023, Martel Lab, Sunnybrook Research Institute

Description: Example code of how to use the project_team to train a model on
classification. This example is performed on the MNIST dataset. This will
perform a Train-Test Split experiment

Input: a working_dir (working directory) to perform the experiment in
Output: in the working directory there will be configs for the manager,
processor, practitioner, and model. Checkpoint saves will be intheir own
folder. Datsets will be saved in individual csv files. Final model weights as a
pth file.
'''

import argparse

import pandas as pd
import numpy as np
from project_team.dt_project.dt_processing import _TensorProcessing
from torchvision import datasets
from sklearn.metrics import accuracy_score

import project_team as proteam
import os
from default_arguements import dt_args, ml_args

from TSN import TSNModel_config, TSNPractitioner_config, \
    TSNModel, TSNPractitioner

r_seed = 20230117
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--working_dir',type=str,
                    default=os.getcwd(),
                    help='The current directory to save models, and configs '
                         'of the experiment')
parser.add_argument('--start_from_checkpoint', action='store_true',
                    help='Choice to load the model from checkpoint folder')
opt = parser.parse_args()

# Prepare data if not already saved and set up
if not os.path.exists(os.path.join(opt.working_dir, 'data','dataset_info.csv')):
    if not os.path.exists(os.path.join(opt.working_dir, 'data')):
        os.mkdir(os.path.join(opt.working_dir, 'data'))

    dataset1 = datasets.MNIST('../data', train=True, download=True)

    dataset2 = datasets.MNIST('../data', train=False)
    all_data = []
    cnt = 0
    for ex in dataset1:
        save_local = os.path.join(opt.working_dir, 'data',
                                  'img_' + str(cnt) + '.png')
        ex[0].save(save_local)
        all_data.append(
            {'img_data': save_local,
             'label': ex[1]}
        )
        cnt += 1
    for ex in dataset2:
        save_local = os.path.join(opt.working_dir, 'data',
                                  'img_' + str(cnt) + '.png')
        ex[0].save(save_local)
        all_data.append(
            {'img_data': save_local,
             'label': ex[1]}
        )
        cnt += 1
    all_data = pd.DataFrame(all_data)
    all_data.to_csv(os.path.join(opt.working_dir, 'data', 'dataset_info.csv'))
    print('Saved all images and data set file to ' + opt.working_dir + '/data')

# Prepare Manager
io_args = {
    'data_csv_location':os.path.join(opt.working_dir, 'dataset_10percent.csv'),
    'inf_data_csv_location': None,
    'val_data_csv_location': None,
    'experiment_name':'MNIST_TSN_TrainTestSplit',
    'project_folder':opt.working_dir,
    'X':'img_data',
    'X_dtype':'PIL png',
    'y':'label',
    'y_dtype':'discrete',
    'y_domain': [_ for _ in range(10)],
    'group_data_by':None,
    'test_size': 0.1,
    'validation_size': 0.1,
    'stratify_by': 'label',
    'r_seed': r_seed
}

io_project_cnfg = proteam.io_project.io_traindeploy_config(**io_args)

manager = proteam.io_project.Pytorch_Manager(
    io_config_input=io_project_cnfg
)

# Prepare Processor
dt_args['one_hot_encode'] = False
dt_args['max_classes'] = 10

dt_project_cnfg = proteam.dt_project.Image_Processor_config(**dt_args)

processor = proteam.dt_project.Image_Processor(
    image_processor_config=dt_project_cnfg
)

# class y_update(_TensorProcessing):
#
#     def __call__(self, ipt):
#         ipt['y'] = [np.array(ipt['y'])[np.newaxis, ...]]
#         return ipt
#
# processor.add_pretransforms([
#     y_update(),
# ])
# Prepare model
mdl_config = TSNModel_config(
    S_x=784,
    S_y=1,
    P=128,
    N_n_units=5000,
    N_ncycle=16
)

mdl = TSNModel(mdl_config)

ml_args['batch_size'] = 128
ml_args['n_epochs'] = 10
ml_args['lr'] = 0.0015
ml_args['batch_epochs'] = 10
# Prepare Practitioner
pt_config = TSNPractitioner_config(**ml_args)

practitioner = TSNPractitioner(config=pt_config, model=mdl,
                               dt_processor=processor,
                               manager=manager)

# Perform Training
manager.prepare_for_experiment()

processor.set_training_data(manager.root)
processor.set_validation_data(manager.root)

errors = practitioner.train_model()

# Perform Inference
manager.prepare_for_inference()

processor.set_inference_data(manager.root)

practitioner.run_inference()

test_results = processor.inference_results

# Evaluate Inference Results
print('Model Accuracy: ' +
      str(accuracy_score(test_results['y'], test_results['pred_y'])))

print('End of MNIST_Classification.py')
