* Directory of the **hyperparameters unique to each method** (e.g., IRM): 

  `./configs/sweep_configs/IRM/base.yaml`

* Directory of the **learning rate, weight decay** of each dataset: /configs/sweep_configs (e.g., GOODArxiv):

  `./configs/sweep_configs/GOODArxiv.yaml`

* Directory of the **training epochs, type of GNN** of an algorithm on a split of a dataset (e.g., GOODCBAS+CIA):

  `./configs/GOOD_configs/GOODCBAS/color/covariate/CIA.yaml`

* Directory of the **number of layers GNN** for a dataset (e.g., GOODCora):

  `./configs/GOOD_configs/GOODCora/base.yaml`

* Where to specify **the directory to save the training checkpoints**:

  in `./GOOD/configs/GOOD_configs/base.yaml`, the `ckpt_root:` term.

* Configurations of **the invariant subgraph generator**:

  in `./GOOD/configs/GOOD_configs/base.yaml`, 

  * `use_inv_edge_mask`: whether to use the invariant subgraph generator or not
  * `train:edge_mask_GNN_lr:` learning rate of the invariant subgraph generator for GAT
  * `train:GCN_edge_mask_GNN_lr:` learning rate of the invariant subgraph generator for GCN