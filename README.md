#  On Redundancy and Diversity in Cell-based Neural Architecture Search

This is the code repository of our ICLR 2022 paper: [On Redundancy and Diversity in Cell-based Neural Architecture Search](https://openreview.net/forum?id=rFJWoYoxrDB).

### Citation
If you find our work to be useful, please cite:

Xingchen Wan, Binxin Ru, Pedro M. Esperança, Zhenguo Li.  On Redundancy and Diversity in Cell-based Neural Architecture Search. 
In *Proceedings of the 10th International Conference on Learning Representations (ICLR)*, 2022.

or in bibtex:

```
@inproceedings{
wan2022on,
title={On Redundancy and Diversity in Cell-based Neural Architecture Search},
author={Xingchen Wan and Binxin Ru and Pedro M Esperan{\c{c}}a and Zhenguo Li},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=rFJWoYoxrDB}
}
```

## Instructions

### Dependencies

We include the dependencies in ```requirements.txt```. We use Anaconda Python 3.7
We run CIFAR-10 experiments on a single NVIDIA V100 GPU. For ImageNet, we use 8x NVIDA V100 GPUs. The training and 
evaluation details can be found in Appendix of the paper.

Please also download the [```autodl``` package](https://github.com/D-X-Y/AutoDL-Projects) and ```natsbench``` package and place them under the root directory.
Make sure they are added to your ```PATH```  as well (```./autodl``` and ```./autodl/lib```)

You also need to download the NAS-Bench-301 and NAS-Bench-201 data files. Since we use the data of the training architectures used to
construct NB301, you also additionally need to download an additional file containing the statistics of these architectures: [```nb_301_v13_lc_iclr_final.zip```](https://figshare.com/articles/dataset/nasbench301_full_data/13286105). This link
is available as of March 2022; if it stops working you may have to contact the NB301 authors.

### Experiments

We use a number of analysis/plotting notebooks to help any interested readers in reproducing the main results of the paper:

1. Reproducing Section 3 (Operation-level results)
   1. Use the scripts available in ```./test/compute_impt_weightss_all_nb{201}/{301}.py``` to precompute the operation importance
   as defined in the paper for top/bottom nb201/301 architectures. Running this script will deposit pre-computed files under a directory ```./data```
   3. Go to ```./notebooks/operation_importance.ipynb``` to generate the results presented in Fig 3 and 4. To see the result
   on successively removing the important/unimportant operations, see ```./notebooks/remove_important_ops.ipynb```.


2. Reproducing Section 4 (Motif-level results)
   1. Run ```./process_all_nb301.py``` to precompute importance info of all training graphs used to construct NB301. This will create a file in 
   ```./data/nb301_evaluated_arch_info.pickle``` (unless you change the flags in the script)
   2. Run ```mine_motifs.py```, which runs the gSpan algorithm. We use the following parameters:
   ```
   python3 -u mine_motifs.py --nb_path={YOUR_PATH_TO_NASBENCH_301_DATA} --min_num_vertices=2 --max_num_vertices=6 --weight_thres=0.001 --normalOnly
   ```
   Note that in general frequent subgraph mining is hard; if you decrease the weight threshold or increase the size of input graph,
   the gSpan algorithm may take a very long time to run. We use the above parameters to ensure efficiency.
   3. Go to ```./notebooks/motif_analysis_gspan.ipynb``` to analyse the frequent motifs

3. Sampling under Prim, Skip and PrimSkip Constraints
   1. Run ```./sample_archs.ipynb``` which randomly samples Prim, Skip and PrimSkip-constrained architectures. The notebook also queries
   the NB301 to obtain the predicted performance.
   2. To reproduce our result that actually trains the architectures (instead of using the predicted performance from NB301), you may save the
   genotypes, and replace ```ss.query``` with ```ss.train```. Go to `./search_spaces/nas301.py` and ```./darts_cnn/{train/eval}_class.py``` to view or modify the training setup.
   The training hyperparameters and other setups used in this paper may be found in the Appendix.
   
## Acknowledgements
The authors thank the maintainers of the following open-sourced repositories:

1. https://github.com/D-X-Y/Awesome-AutoDL
2. https://github.com/D-X-Y/AutoDL-Projects
3. https://github.com/D-X-Y/NATS-Bench
4. https://github.com/quark0/darts
5. https://github.com/automl/nasbench301
