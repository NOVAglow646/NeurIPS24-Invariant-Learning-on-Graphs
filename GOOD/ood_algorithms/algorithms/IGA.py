"""
Implementation of the IGA algorithm from `"Out-of-Distribution Generalization via Risk Extrapolation (REx)"
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
class IGA(BaseOODAlg):
    r"""
    Implementation of the IGA algorithm from `"Out-of-Distribution Generalization via Risk Extrapolation (REx)"
    <http://proceedings.mlr.press/v139/krueger21a.html>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(IGA, self).__init__(config)

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on IGA algorithm

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
            loss based on IGA algorithm

        """
        model=kwargs['model']
        gradients_list = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            if loss[env_idx].shape[0] > 0:
                normalized_loss = loss[env_idx].sum() / mask[env_idx].sum()
                normalized_loss.backward(retain_graph=True)  # 再次计算梯度

                # 收集梯度
                gradients = []
                for param in model.parameters():
                    if param.grad is not None:
                        # 使用param.grad.data来获取梯度数据
                        gradients.append(param.grad.data.view(-1))
                
                # 组合所有梯度
                env_gradients = torch.cat(gradients)
                gradients_list.append(env_gradients)

                # 清除梯度以便下一次计算
                model.zero_grad()

        # 计算每个环境的梯度的方差
        gradients_stack = torch.stack(gradients_list)
        gradients_var = torch.var(gradients_stack, dim=0)

        # 现在gradients_var是每个参数的梯度方差，你可以对它们求和或者取平均值
        # 以及添加到你的总损失中
        spec_loss = config.ood.ood_param * gradients_var.mean()  # 或者使用.sum()来取得总方差

        if torch.isnan(spec_loss):
            spec_loss = 0
        mean_loss = loss.sum() / mask.sum()
        loss = spec_loss + mean_loss
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss

        # Add this at the end of the function before returning the loss
        '''if  config.train.batch_id>config.train.num_batches-2 and config.train.epoch%3==0:
            #print(F'#in# {K}')
            save_path = '/data1/qxwang/codes/GOOD/visualization/IGA_t-SNE'
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'IGA_{config.dataset.dataset_name}_{config.dataset.shift_type}_{config.dataset.domain}_tsne_epoch_{config.train.epoch}.png')
            if int(torch.max(data.y))>8: # if this dataset too many classes
                self.visualize_tsne(self.rep, data.y, data.env_id, file_path, drop_classes=True, num_classes_to_visualize=8)
            else:
                self.visualize_tsne(self.rep, data.y, data.env_id, file_path)'''

        return loss
