"""
Implementation of the baseline ERM
"""
import torch
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
import os

@register.ood_alg_register
class ERM(BaseOODAlg):
    r"""
    Implementation of the baseline ERM

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(ERM, self).__init__(config)
        self.rep=None

    '''def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        pred, self.rep=model_output
        return pred'''

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:

        
        # Add this at the end of the function before returning the loss
        '''if config.train.epoch<40 and config.train.batch_id>config.train.num_batches-2 and config.train.epoch%3==0:
            save_path = '/data1/qxwang/codes/GOOD/visualization/ERM_t-SNE'
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'ERM_{config.dataset.dataset_name}_tsne_epoch_{config.train.epoch}.png')
            self.visualize_tsne(self.rep, data.y, data.env_id, file_path, drop_classes=True, num_classes_to_visualize=8)'''

        self.mean_loss = loss.sum() / mask.sum()
        if self.config.ood.extra_param[0]:
            writer=kwargs['writer']
            var, dis=self.var_dis(self.rep, data.y)
            print(f'#in# {var/config.dataset.num_classes} {dis/config.dataset.num_classes**2}')
            return self.mean_loss, var/config.dataset.num_classes, dis/config.dataset.num_classes**2
        return self.mean_loss
