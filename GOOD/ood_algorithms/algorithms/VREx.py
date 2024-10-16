"""
Implementation of the VREx algorithm from `"Out-of-Distribution Generalization via Risk Extrapolation (REx)"
<http://proceedings.mlr.press/v139/krueger21a.html>`_ paper
"""
import torch
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch
import os

@register.ood_alg_register
class VREx(BaseOODAlg):
    r"""
    Implementation of the VREx algorithm from `"Out-of-Distribution Generalization via Risk Extrapolation (REx)"
    <http://proceedings.mlr.press/v139/krueger21a.html>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(VREx, self).__init__(config)

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on VREx algorithm

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
            loss based on VREx algorithm

        """
        loss_list = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            if loss[env_idx].shape[0] > 0:
                loss_list.append(loss[env_idx].sum() / mask[env_idx].sum())
        spec_loss = config.ood.ood_param * torch.var(torch.stack(loss_list))
        if torch.isnan(spec_loss):
            spec_loss = 0
        mean_loss = loss.sum() / mask.sum()
        loss = spec_loss + mean_loss
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss

        # Add this at the end of the function before returning the loss
        '''if  config.train.batch_id>config.train.num_batches-2 and config.train.epoch%3==0:
            #print(F'#in# {K}')
            save_path = '/data1/qxwang/codes/GOOD/visualization/VREx_t-SNE'
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'VREx_{config.dataset.dataset_name}_{config.dataset.shift_type}_{config.dataset.domain}_tsne_epoch_{config.train.epoch}.png')
            if int(torch.max(data.y))>8: # if this dataset too many classes
                self.visualize_tsne(self.rep, data.y, data.env_id, file_path, drop_classes=True, num_classes_to_visualize=8)
            else:
                self.visualize_tsne(self.rep, data.y, data.env_id, file_path)'''

        return loss
