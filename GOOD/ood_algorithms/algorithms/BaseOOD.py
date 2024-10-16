"""
Base class for OOD algorithms
"""
from abc import ABC
from torch import Tensor
from torch_geometric.data import Batch
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from typing import Tuple
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
import torch

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np

class BaseOODAlg(ABC):
    r"""
    Base class for OOD algorithms

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(BaseOODAlg, self).__init__()
        self.optimizer: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.model: torch.nn.Module = None


        self.mean_loss = None
        self.spec_loss = None
        self.stage = 0

        
        #added by AXZ
        self.rep=None
        self.config=config


    def visualize_tsne(self, representations, labels, env_ids, file_path, drop_classes=False, num_classes_to_visualize=8):
        """
        Visualize representations using t-SNE.
        By AXZ

        Args:
            representations (Tensor): The representations to be visualized. It can also be original node feature x.
            labels (Tensor): Class labels corresponding to each representation.
            env_ids (Tensor): Environment IDs corresponding to each representation.
            file_path (str): The path where to save the t-SNE visualization.
        """

        # Convert tensor to numpy array
        representations = representations.cpu().detach().numpy()
        labels = labels.cpu().numpy()
        env_ids = env_ids.cpu().numpy()

        # Get unique labels and randomly select a subset
        unique_labels = np.unique(labels)
        if drop_classes:
            selected_labels = np.random.choice(unique_labels, num_classes_to_visualize, replace=False)
            # Filter representations, labels, and environment IDs to include only the selected classes
            mask = np.isin(labels, selected_labels)
            representations = representations[mask]
            labels = labels[mask]
            env_ids = env_ids[mask]

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(representations)

        # Define different markers for different classes
        markers = ['o', '^','p','*', 's','D','X','8', 'h',',','<',    'v',   'H',  'd', '>','P' ]
        
        # Get unique labels and environments
        unique_env_ids = np.unique(env_ids)

        # Define different colors for different environments
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_env_ids)))

        # Plotting
        plt.figure(figsize=(15, 9))

        # Loop over each class and each environment to plot separately with unique color and marker
        if drop_classes:
            lb=selected_labels
        else:
            lb=unique_labels
        for i, label in enumerate(lb):
            for j, env_id in enumerate(unique_env_ids):
                mask = (labels == label) & (env_ids == env_id)
                plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], marker=markers[i % len(markers)], color=colors[j], alpha=0.5)

        #plt.title('t-SNE visualization of representations')
        # Adding legends separately for classes and environments
        class_legend = [plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', label=f'Class {unique_labels[i]}', markersize=10) for i in range(len(lb))]
        env_legend = [plt.Line2D([0], [0], marker='o', color=colors[i], label=f'Env {unique_env_ids[i]}', markersize=10) for i in range(len(unique_env_ids))]
        
        # First legend for class
        #legend1 = plt.legend(handles=class_legend, loc=(0.9, 0.7))
        #plt.gca().add_artist(legend1)
        
        # Second legend for environment
        #plt.legend(handles=env_legend, loc=(0.9, 0.0))

        # Save the plot
        plt.savefig(file_path)
        plt.close()  # Close the plot to free up memory


    def stage_control(self, config):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:
        r"""
        Set input data format and preparations

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns:
            - data (Batch) - Processed input data.
            - targets (Tensor) - Processed input labels.
            - mask (Tensor) - Processed NAN masks for data formats.
            - node_norm (Tensor) - Processed node weights for normalization.

        """
        return data, targets, mask, node_norm

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        if self.config.model.model_name=='CIAGCN':
            model_output, self.rep=model_output
        return model_output

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor, config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate prediction loss without any special OOD constrains

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
        #print(F'#in# pred.shape={raw_pred.shape}') # pred.shape=torch.Size([9342, 70])
        #print(F'#in# mask.sum()={mask.sum()}') #mask.sum()=3467
        #print(F'#in# targets={targets}')
        #print(F'#in# targets.min {targets.min()}')
        loss = config.metric.loss_func(raw_pred, targets.long(), reduction='none') * mask
        
        #print(F'#in# node_norm={node_norm.shape}') 
        #print(F'#in# mask.sum()={mask.sum()}')
        loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss
        return loss

    def var_dis(self, rep, Y):
        '''
        According to Kothapalli et al 2023, lower CDNV should indicate better classification performance
        '''
        var=0.
        dis=0.
        for c1 in range(self.config.dataset.num_classes):
            c1_idx=torch.where(Y==c1)[0]
            rep_c1=rep[c1_idx]
            if rep_c1.numel()==0:
                continue
            var1, mean1=torch.var_mean(rep_c1, dim=0, correction=0)
            for c2 in range(c1+1, self.config.dataset.num_classes):
                c2_idx=torch.where(Y==c2)[0]
                rep_c2=rep[c2_idx]
                if rep_c2.numel()==0:
                    continue
                var2, mean2=torch.var_mean(rep_c2, dim=0, correction=0)
                var+=torch.mean(var1)
                dis+=torch.norm(mean1-mean2)
        return var, dis

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns (Tensor):
            processed loss

        """

        self.mean_loss = loss.sum() / mask.sum()
        return self.mean_loss

    def set_up(self, model: torch.nn.Module, config: Union[CommonArgs, Munch], **kwargs):
        r"""
        Training setup of optimizer and scheduler

        Args:
            model (torch.nn.Module): model for setup
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.train.lr`, :obj:`config.metric`, :obj:`config.train.mile_stones`)

        Returns:
            None

        """
        self.model: torch.nn.Module = model
        if config.use_inv_edge_mask:
            self.edge_mask_GNN=kwargs['edge_mask_GNN']
            #print('#in# Registering the subgraph extractor in the optimizer')
            #print(f'#in# config.train.lr {config.train.lr}')
            #print(f'#in# config.train.edge_mask_GNN_lr {config.train.edge_mask_GNN_lr}')
            if config.model.model_name in ['CIAGAT','GAT']:
                edge_mask_lr = config.train.edge_mask_GNN_lr 
            elif config.model.model_name in ['CIAGCN','GCN']:
                edge_mask_lr = config.train.GCN_edge_mask_GNN_lr
                
            self.optimizer = torch.optim.Adam([{'params':self.model.parameters(), 'lr':config.train.lr}, 
                                               {'params':self.edge_mask_GNN.parameters(), 'lr':edge_mask_lr}],
                                          weight_decay=config.train.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr,
                                          weight_decay=config.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.train.mile_stones,
                                                              gamma=0.1)
        
    def backward(self, loss):
        r"""
        Gradient backward process and parameter update.

        Args:
            loss: target loss
        """
        loss.backward()
        self.optimizer.step()
