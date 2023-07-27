# Measuring and Modeling Uncertainty Degree for Monocular Depth Estimation

[[arXiv](https://arxiv.org/abs/2307.09929)] 

## Introduction

This is a PyTorch implementation of *Measuring and Modeling Uncertainty Degree for Monocular Depth Estimation*.

## Training

### Data preparation

Please change the path to the NYU V2 zip file in `utils/options.py` (check out [DenseDepth](https://github.com/ialhashim/DenseDepth) to download the zipfile):
```python
DEFAULTS = {'nyu_data_path':
                {'your-pc-name-here': 'your-data-path-here'},
            ...}
```

Please chenge the path to the KITTI dataset in `utils/kitti_data.py`:
```python
class KittiDefaultArg(object):
    def __init__(self, opts):
        self.dataset = 'kitti'
        self.filenames_file = 'utils/kitti/kitti_eigen_train_files_with_gt_new.txt'
        self.filenames_file_eval = 'utils/kitti/kitti_eigen_test_files_with_gt.txt'
        self.data_path = 'MODIFY_ME/kitti_data'
        self.gt_path = 'MODIFY_ME/data_depth_annotated'

        self.data_path_eval = 'MODIFY_ME/kitti_data'
        self.gt_path_eval = 'MODIFY_ME/data_depth_annotated'

```

### Requirements

```
pip install -r requirements.txt
```

### Training 

There are a few options to train an MDE network, you can choose between:

**dataset**: `nyu`, `kitti`

**encoder**: `resnet`, `densenet`, `swin`, `vit`, `efficientnet`

**decoder**: `simple`, `bts`

**reg_mode**: `direct`, `lin_cls`

**reg_supervision**: `regression_l1`, `none`

**prob_supervision**: `soft_label`, `none` 

**uncert_supervision**: `error_uncertainty_ranking`, `error_uncertainty_ranking_noclamp`, `error_uncertainty_l1`, `none`


You can choose between these options by adding extra arguments, for example:
```
python train.py \
  --dataset nyu \
  --encoder swin \
  --reg_mode lin_cls \
  --reg_supervision regression_l1 \
  --prob_supervision soft_label \
  --uncert_supervision error_uncertainty_ranking
```
  
### Evaluation

To evaluate the accuracy and uncertainty degree of a trained model, run `eval.py` and add the path to the model checkpoint folder as an extra artument.
```
python eval.py Res_Sim_Lin_L-1_Non_Eur_kitti_2023_02_28-22:42:51
```

## Citation

```
@article{xiang2023measuring,
  title={Measuring and Modeling Uncertainty Degree for Monocular Depth Estimation},
  author={Xiang, Mochu and Zhang, Jing and Barnes, Nick and Dai, Yuchao},
  journal={arXiv preprint arXiv:2307.09929},
  year={2023}
}
```