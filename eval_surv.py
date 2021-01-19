from __future__ import print_function

import argparse
import pdb
import os
import math
import sys

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train, eval_model
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from timeit import default_timer as timer


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    val_cindex = []
    folds = np.arange(start, end)

    for i in folds:
        start = timer()
        seed_torch(args.seed)

        train_dataset, val_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)
        
        if 'omic' in args.mode:
            args.omic_input_dim = train_dataset.genomic_features.shape[1]
            print("Genomic Dimension", args.omic_input_dim)

        val_latest, cindex_latest = eval_model(datasets, i, args)
        val_cindex.append(cindex_latest)

        #write results to pkl
        save_pkl(os.path.join(args.results_dir, 'split_val_{}_results.pkl'.format(i)), val_latest)
        end = timer()
        print('Fold %d Time: %f seconds' % (i, end - start))

    if len(folds) != args.k: save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else: save_name = 'summary.csv'
    results_df = pd.DataFrame({'folds': folds, 'val_cindex': val_cindex})
    results_df.to_csv(os.path.join(args.results_dir, 'summary.csv'))

# Training settings
parser = argparse.ArgumentParser(description='Configurations for MMF Training')
parser.add_argument('--data_root_dir', type=str, default='/media/ssd1/pan-cancer', help='data directory')
parser.add_argument('--which_splits', type=str, default='5foldcv', help='Path to splits directory.')
parser.add_argument('--split_dir', type=str, help='Set of splits to use for each cancer type.')
parser.add_argument('--mode', type=str, default='omic')
parser.add_argument('--model_type', type=str, default='clam', help='type of model (attention_mil | max_net | mm_attention_mil)')

parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight', type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None, help='instance-level clustering loss function (default: None)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Reg Type (default: None)')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='Regularization Strength')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='enable weighted sampling')
parser.add_argument('--model_size_wsi', type=str, default='small', help='Size of AMIL model.')
parser.add_argument('--model_size_omic', type=str, default='small', help='Size of SNN Model.')
parser.add_argument('--gc', type=int, default=1, help='gradient accumulation step')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
parser.add_argument('--gate_path', action='store_true', default=False, help='Enable feature gating in MMF layer.')
parser.add_argument('--gate_omic', action='store_true', default=False, help='Enable feature gating in MMF layer.')
parser.add_argument('--fusion', type=str, default='tensor', help='Which fusion mechanism to use.')
parser.add_argument('--overwrite', action='store_true', default=False, help='Current experiment results already exists. Redo?')
parser.add_argument('--apply_mad', action='store_true', default=True, help='Use genes with median absolute deviation.')
parser.add_argument('--task', type=str, default='survival', help='Which task.')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Creates Custom Experiment Code
exp_code = '_'.join(args.split_dir.split('_')[:2])
dataset_path = 'dataset_csv'
param_code = ''

if args.model_type == 'attention_mil':
  param_code += 'WSI'
elif args.model_type == 'max_net':
  param_code += 'SNN'
elif args.model_type == 'mm_attention_mil' and args.fusion == 'tensor':
  param_code += 'MMF'
else:
  raise NotImplementedError

if 'small' in args.model_size_omic:
  param_code += 'sm'

param_code += '_%s' % args.bag_loss

if 'mm_' in args.model_type:
  param_code += '_g'
  if args.gate_path:
    param_code += '1'
  else:
    param_code += '0'

  if args.gate_omic:
    param_code += '1'
  else:
    param_code += '0'

param_code += '_a%s' % str(args.alpha_surv)

if args.lr != 2e-4:
  param_code += '_lr%s' % format(args.lr, '.0e')

if args.reg_type != 'None':
  param_code += '_reg%s' % format(args.lambda_reg, '.0e')

param_code += '_%s' % args.which_splits.split("_")[0]

if args.gc != 1:
  param_code += '_gc%s' % str(args.gc)

if args.apply_mad:
  param_code += '_mad'
  #dataset_path += '_mad'
  
args.exp_code = exp_code + "_" + param_code

### task
if args.task == 'survival':
  args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
print("Experiment Name:", exp_code)


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'inst_loss': args.inst_loss,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt}

print('\nLoad Dataset')
if args.task == 'tcga_blca_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv' % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_bladder_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_brca_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_breast_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_coadread_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_coadread_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_gbmlgg_survival':
    args.n_classes = 4
    dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/tcga_gbmlgg_all.csv' % dataset_path,
                                           mode = args.mode,
                                           data_dir={'ASTR': os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                                     'AASTR': os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                                     'ODG': os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                                     'OAST': os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                                     'AOAST': os.path.join(args.data_root_dir,'tcga_lgg_20x_features'),
                                                     'GBM': os.path.join(args.data_root_dir,'tcga_gbm_20x_features'),},
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_hnsc_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_hnsc_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_kirc_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_kirp_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_lihc_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_liver_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_luad_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_lung_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_lusc_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_lung_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_paad_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_pancreas_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_skcm_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_skin_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_stad_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_stomach_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
elif args.task == 'tcga_ucec_survival':
  args.n_classes = 4
  proj = '_'.join(args.task.split('_')[:2])
  dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all.csv'  % (dataset_path, proj),
                                           mode = args.mode,
                                           data_dir= os.path.join(args.data_root_dir, 'tcga_endometrial_20x_features'),
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
else:
  raise NotImplementedError

if isinstance(dataset, Generic_MIL_Survival_Dataset):
    args.task_type ='survival'
else:
    raise NotImplementedError

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

### GET RID OF WHICH_SPLITS IF U WANT TO MAKE THE RESULTS FOLDER LESS CLUTTERRED
args.results_dir = os.path.join(args.results_dir, args.which_splits, param_code, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
  print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
  sys.exit()

if args.split_dir is None:
    args.split_dir = os.path.join('./splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)

print("split_dir", args.split_dir)

assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":

    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
