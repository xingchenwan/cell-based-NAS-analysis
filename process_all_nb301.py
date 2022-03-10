# Process all the evaluated architectures in the NB301 dataset
import argparse
import os
import json
import networkx as nx
import numpy as np
from utils import archinfo2genotype, genotype2networkx
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser('Data processor for NB301 for all methods')
parser.add_argument('-src', '--source_path', required=True)
parser.add_argument('-dst', '--destination_path', default='./data/')
parser.add_argument('--edge_attributed', action='store_true', help='whether the DAGs generated should be edge attributed'
                                                                   'or node attributed')

args = parser.parse_args()
print(vars(args))

if not os.path.exists(args.destination_path):
    os.makedirs(args.destination_path)

available_methods = os.listdir(args.source_path)

all_archs = []
for method in tqdm(available_methods):
    file_path = os.path.join(args.source_path, method)
    n_archs = len(os.listdir(file_path))
    method_files = os.listdir(file_path)
    archs_to_sample = np.arange(n_archs)

    for j, arch_idx in tqdm(enumerate(archs_to_sample)):
        file_selected = method_files[arch_idx]
        arch_path = os.path.join(file_path, file_selected)
        arch_data = json.load(open(arch_path, 'r'))
        arch_data['genotype'] = archinfo2genotype(arch_data)
        # also store the method that produces the method
        arch_data['method'] = method
        all_archs.append(arch_data)

# select the top "args.top_pct" percent architectures based on their validation accuracy
# top_n = np.round(n_arch_data * args.top_pct).astype(np.int)
# top_indices = np.argpartition(all_accs, -top_n)[-top_n:]
# top_archs = [all_res[i] for i in top_indices]
pickle.dump(all_archs, open(os.path.join(args.destination_path, f'nb301_evaluated_arch_info.pickle'), 'wb'))
