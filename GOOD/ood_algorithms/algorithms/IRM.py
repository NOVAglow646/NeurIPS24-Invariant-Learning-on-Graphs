"""
Implementation of the IRM algorithm from `"Invariant Risk Minimization"
<https://arxiv.org/abs/1907.02893>`_ paper
"""
import torch
from torch.autograd import grad
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np

@register.ood_alg_register
class IRM(BaseOODAlg):
    r"""
    Implementation of the IRM algorithm from `"Invariant Risk Minimization"
    <https://arxiv.org/abs/1907.02893>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(IRM, self).__init__(config)
        self.dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(config.device)
        self.rep=None


    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; apply the linear classifier

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions with the linear classifier applied

        """
        if not isinstance(model_output, tuple):
            raw_pred = self.dummy_w * model_output
        else:
            pred, self.rep=model_output
            raw_pred=self.dummy_w*pred
        return raw_pred

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on IRM algorithm

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
            loss with IRM penalty

        """
        spec_loss_list = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            #print(F'#in# loss.shape={loss.shape}')
            #print(F'#in# loss[env_idx].shape={loss[env_idx].shape}')
            if loss[env_idx].shape[0] > 0:
                grad_all = torch.sum(
                    grad(loss[env_idx].sum() / mask[env_idx].sum(), self.dummy_w, create_graph=True)[0].pow(2))
                spec_loss_list.append(grad_all)
        spec_loss = config.ood.ood_param * sum(spec_loss_list) / len(spec_loss_list)
        if torch.isnan(spec_loss):
            spec_loss = 0
        mean_loss = loss.sum() / mask.sum()
        loss = spec_loss + mean_loss
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss

        # Add this at the end of the function before returning the loss
        '''if config.train.epoch>90 and config.train.batch_id>config.train.num_batches-2:
            save_path = '/data1/qxwang/codes/GOOD/visualization/IRM_t-SNE'
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'IRM_tsne_epoch_{config.train.epoch}.png')
            self.visualize_tsne(self.rep, data.y, data.env_id, file_path)'''

        return loss
