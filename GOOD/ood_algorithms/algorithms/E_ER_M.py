"""
Implementation of the E_ER_M algorithm from `"Handling Distribution Shifts on Graphs: An Invariance Perspective" <https://arxiv.org/abs/2202.02466>`_ paper
"""
import torch
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.train import at_stage
from GOOD.utils.initial import reset_random_seed

import os

@register.ood_alg_register
class E_ER_M(BaseOODAlg):
    r"""
    Implementation of the E_ER_M algorithm from `"Handling Distribution Shifts on Graphs: An Invariance Perspective"
    <https://arxiv.org/abs/2202.02466>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(E_ER_M, self).__init__(config)

    def stage_control(self, config: Union[CommonArgs, Munch]):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1
            self.optimizer = torch.optim.Adam(self.model.gnn.parameters(), lr=config.train.lr,
                                              weight_decay=config.train.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=config.train.mile_stones,
                                                                  gamma=0.1)

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor, config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func: Accuracy}
                                   })


        Returns (Tensor):
            cross entropy loss

        """
        assert config.model.model_level == 'node'
        return raw_pred

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on E_ER_M algorithm

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   dataset: {num_envs: int(10)},
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns (Tensor):
            loss based on E_ER_M algorithm

        """
        '''beta = 10 * config.ood.ood_param * config.train.epoch / config.train.max_epoch \
               + config.ood.ood_param * (1 - config.train.epoch / config.train.max_epoch)'''
        CIA_loss, Mean = loss
        loss = Mean + config.ood.ood_param * CIA_loss
        self.mean_loss = Mean
        self.spec_loss = CIA_loss

        # Add this at the end of the function before returning the loss
        '''if config.train.epoch>90 and config.train.batch_id>config.train.num_batches-2 and config.train.epoch % 10==0:
            save_path = '/data1/qxwang/codes/GOOD/visualization/EERM_t-SNE'
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'EERM_{config.dataset.dataset_name}_gt-env-partition_tsne_epoch_{config.train.epoch}.png')
            self.visualize_tsne(self.rep, data.y, data.env_id, file_path)'''

        return loss
