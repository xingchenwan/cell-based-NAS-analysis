# train a genotype from scratch using train or eval mode.

import torch
torch.cuda.empty_cache()
# import tensorflow as tf
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)   # tensorflow is not used in pipeline but could be required during init
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from search_spaces.nas301 import NASBench301, genotype2str
import pickle
import os
import numpy as np
import argparse
from search_spaces.nas301 import Genotype       # this is required to unpickle the list of genotypes, even it appears to be an unused import
# import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-fp', '--file_path', required=True, help='location to the saved genotypes. Should be a list of genotypes')
parser.add_argument('-ip', '--image_path', default=None)
parser.add_argument('-m', '--mode', choices=['train', 'eval', 'eval_extend', 'eval_test'], default='train')
parser.add_argument('--n_train_archs', type=int, default=20, )
parser.add_argument('--start_idx', type=int, default=None)
parser.add_argument('--end_idx', type=int, default=None)
parser.add_argument('-sp', '--save_path', default='./output/train_genotypes/')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=None)
parser.add_argument('--as_list', action='store_true')
parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'imagenet'], default='cifar10')
parser.add_argument('--no_mixup', action='store_true')
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--retrain_exist', action='store_true', help='whether to retrain genotypes previously trained.')

args = parser.parse_args()
# time_string = datetime.datetime.now()
# time_string = time_string.strftime('%Y%m%d_%H%M%S')
options = vars(args)
print(options)

if args.image_path is None:
    home_dir = os.path.expanduser('~')
    if 'xwan' in home_dir:      # local
        print('Local')
        img_path = '/media/xwan/HDD2/NASDatasets/'
    elif 'nfs' in home_dir:         # rapid
        print('In Rapid')
        img_path = '/nfs/home/xingchenw/NASDatasets/'
    else:
        raise ValueError(f'Novel environment. The home directory is {home_dir}')
else:
    img_path = args.image_path

# args.save_path = os.path.join(args.save_path, time_string)
# if args.dataset != 'cifar10':
#     args.save_path += f'{args.dataset}'
if args.experiment_name is not None:
    args.save_path += args.experiment_name
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
option_file = open(args.save_path + "/command.txt", "w+")
option_file.write(str(options))
option_file.close()

genotypes = pickle.load(open(args.file_path, 'rb'))
if isinstance(genotypes, dict):
    assert 'genotypes' in genotypes.keys()
    assert 'names' in genotypes.keys()
    genotypes, names = genotypes['genotypes'], genotypes['names']
else:
    names = None

if args.start_idx is not None and args.end_idx is not None:
    start_idx = max(0, args.start_idx)
    end_idx = min(len(genotypes), args.end_idx)
    genotypes = genotypes[start_idx:end_idx]
    if names is not None:
        names = names[start_idx:end_idx]

ss = NASBench301(file_path=None, negative=False, log_scale=False)       # nb301 in API-less mode
ss.image_path = img_path

if args.as_list:
    if args.seed is not None:
        np.random.RandomState(args.seed).shuffle(genotypes)     # shuffle the genotype order in the list
    else:
        np.random.shuffle(genotypes)

    if args.dataset != 'cifar10': save_path = os.path.join(args.save_path, 'as_list' + f'_{args.dataset}')
    else:save_path = os.path.join(args.save_path, 'as_list')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if args.mode == 'train': stats, model = ss.train(genotypes, seed=args.seed, gpu_id=args.gpu_id)
    if args.mode == 'eval': stats, model = ss.evaluate_list(genotypes, seed=args.seed, gpu_id=args.gpu_id, dataset=args.dataset, mixup=not args.no_mixup)
    elif args.mode == 'eval_extend': stats, model = ss.evaluate_list(genotypes, seed=args.seed, epochs=1500, gpu_id=args.gpu_id, dataset=args.dataset, mixup=not args.no_mixup)
    else: raise ValueError()
    stats_to_save = {
        'genotype': genotypes,
        'retrain_stats': stats,
        # 'pred_stats': pred_accs
    }
    pickle.dump(stats_to_save, open(os.path.join(save_path, 'stats.pickle'), 'wb'))
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))

else:
    for i, genotype in enumerate(genotypes):
        if names is None: genotype_str = genotype2str(genotype, flatten=True)
        else: genotype_str = names[i]
        print(f'Starting genotype {i} / {len(genotypes)}: Genotype String/Name = {genotype_str}')
        if args.dataset != 'cifar10':  save_path = os.path.join(args.save_path, genotype_str+f'_{args.dataset}')
        else: save_path = os.path.join(args.save_path, genotype_str)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            if os.path.exists(os.path.join(save_path, 'model.pt')) and not args.retrain_exist:
                print(f'Genotype {genotype_str} is already trained with model saved at {save_path}. Skipped')
                continue

        if args.mode == 'train': stats, model = ss.train(genotype, seed=args.seed, gpu_id=args.gpu_id, mixup=not args.no_mixup)
        elif args.mode == 'eval': stats, model = ss.evaluate(genotype, seed=args.seed,  gpu_id=args.gpu_id, dataset=args.dataset, mixup=not args.no_mixup, epochs=250 if args.dataset == 'imagenet' else 600)
        elif args.mode == 'eval_extend': stats, model = ss.evaluate(genotype, seed=args.seed, epochs=1500,  gpu_id=args.gpu_id, dataset=args.dataset, mixup=not args.no_mixup)
        elif args.mode == 'eval_test': stats, model =ss.evaluate(genotype, seed=args.seed,  gpu_id=args.gpu_id, dataset=args.dataset, mixup=not args.no_mixup, epochs=1, test_mode=True)
        else: raise ValueError()
        # evaluate the genotype prediction for multiple times to obtain uncertainty estimation
        # pred_accs = [ss.query(genotype, with_noise=True) for _ in range(10)]
        stats_to_save = {
            'genotype': genotype,
            'genotype_str': genotype_str,
            'retrain_stats': stats,
            # 'pred_stats': pred_accs
        }
        pickle.dump(stats_to_save, open(os.path.join(save_path, 'stats.pickle'), 'wb'))
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
