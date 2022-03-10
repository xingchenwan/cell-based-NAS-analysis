# Process the NB301 dataset
import argparse
import os
import json
import networkx as nx
import numpy as np
from utils import archinfo2genotype, genotype2networkx
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser('Data processor for NB301')
parser.add_argument('-src', '--source_path', required=True)
parser.add_argument('-dst', '--destination_path', default='./data/')
parser.add_argument('-m', '--method', default='rs')
parser.add_argument('-top', '--top_pct', type=float, default=0.05, help='designate "-top" percent of the architectures as'
                                                                        'the good architectures.')
parser.add_argument('--max_arch_per_method', type=int, default=5000, help='maximum number of archs to sample from each'
                                                                          'method')
parser.add_argument('--edge_attributed', action='store_true', help='whether the DAGs generated should be edge attributed'
                                                                   'or node attributed')
parser.add_argument('--sample_random', action='store_true')

args = parser.parse_args()
print(vars(args))

if not os.path.exists(args.destination_path):
    os.makedirs(args.destination_path)

available_methods = os.listdir(args.source_path)
assert args.method in available_methods, f'{args.method} is not an available method!'

all_res = []
all_accs = []
file_path = os.path.join(args.source_path, args.method)
n_archs = len(os.listdir(file_path))
method_files = os.listdir(file_path)
if args.max_arch_per_method is None:
    archs_to_sample = np.arange(n_archs)
else:
    archs_to_sample = np.random.choice(n_archs, min(n_archs, args.max_arch_per_method), replace=False)
for i, arch_idx in tqdm(enumerate(archs_to_sample)):
    file_selected = method_files[arch_idx]
    arch_path = os.path.join(file_path, file_selected)
    arch_data = json.load(open(arch_path, 'r'))
    arch_data['genotype'] = archinfo2genotype(arch_data)
    arch_data['graph'] = genotype2networkx(arch_data['genotype'], edge_attributed=args.edge_attributed,
                                           return_joint_graph=True)
    all_res.append(arch_data)
    all_accs.append(arch_data['info'][0]['val_accuracy'])

all_accs = np.array(all_accs)
n_arch_data = len(all_res)
# select the top "args.top_pct" percent architectures based on their validation accuracy
top_n = np.round(n_arch_data * args.top_pct).astype(np.int)
top_indices = np.argpartition(all_accs, -top_n)[-top_n:]
top_archs = [all_res[i] for i in top_indices]
pickle.dump(top_archs, open(os.path.join(args.destination_path, f'nb301_{args.method}_top_{args.top_pct}_archs.pickle'), 'wb'))

if args.sample_random:
    # select the same number of archs in the random search group for uniformly sampled archs
    n_rand_archs = len(os.listdir(os.path.join(args.source_path, 'rs')))
    random_arch_idices = np.random.choice(n_rand_archs, min(n_rand_archs, n_arch_data))
    random_arch_files = os.listdir(os.path.join(args.source_path, 'rs'))
    rand_res = []
    for random_arch_index in tqdm(random_arch_idices):
        file_selected = random_arch_files[random_arch_index]
        arch_path = os.path.join(args.source_path, 'rs', file_selected)
        arch_data = json.load(open(arch_path, 'r'))
        arch_data['genotype'] = archinfo2genotype(arch_data)
        arch_data['graph'] = genotype2networkx(arch_data['genotype'], edge_attributed=args.edge_attributed,
                                               return_joint_graph=True)
        rand_res.append(arch_data)
    pickle.dump(rand_res, open(os.path.join(args.destination_path, f'nb301_random_archs.pickle'), 'wb'))
