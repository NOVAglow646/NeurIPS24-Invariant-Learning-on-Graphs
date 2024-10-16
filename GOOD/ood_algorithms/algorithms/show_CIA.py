"""
Implementation of the CIA algorithm from `"Invariant Risk Minimization"
<https://arxiv.org/abs/1907.02893>`_ paper
"""
import torch
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np

import torch_geometric.utils

import time

@register.ood_alg_register
class show_CIA(BaseOODAlg):
    r"""
    CIA: Cross-domain Intra-class Alignment (version 1: use off-the-shelf env partition)

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(show_CIA, self).__init__(config)
        self.rep=None

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        pred, self.rep=model_output
        return pred

    def cal_num_neighbours_with_same_class(self, A, Y):
        '''compute the number of the node with the same class as each node'''
        num_nodes=A.shape[0] # N
        yrep=Y.repeat(num_nodes,1)
        i=torch.arange(num_nodes)
        return torch.sum((A[i] > 0) & (yrep == Y[i].unsqueeze(1)), dim=1).float()-1. # dim=[N]
    
    def cal_Ak(self, A, k):
        '''compute A^k, where A^k_{i,j}=s if i can reach j in s steps 
        (minimal step), s<=k, otherwise=inf'''
        N = A.shape[0]
        # Initialize a matrix to store the minimum number of hops, start with infinity
        Ak = torch.full((N, N), float('inf'), device=self.config.device)   
        # Set the direct edges to 1 hop
        Ak[A == 1] = 1
        # set self hop=0
        torch.diagonal(Ak).fill_(0)
        # Temporary matrix to store paths found in each iteration
        temp_A = A.clone()
        # Find the shortest paths iteratively
        for step in range(1, k):
            temp_A = torch.mm(A, temp_A)
            # If a shorter path is found in this iteration, update the result Ak
            Ak[(temp_A > 0) & (temp_A < Ak)] = step + 1
        #print(f'#in# Ak {Ak}')
        return Ak

    def get_Ak_c(self, Ak, c, Y):
        '''return the sub-adjacency matrix of class c
        Y: dim=[N], labels of the whole graph
        c: int'''
        y_id=torch.where(Y==c)[0]
        row_id=y_id.unsqueeze(1).repeat(1,torch.numel(y_id))
        col_id=y_id.repeat(torch.numel(y_id),1)
        return Ak[row_id, col_id]

    def adjacent_edge_index(self, Ak_c):
        '''return the edge index of the node pairs within k hops of class c, 
        dim=[2, |E_c|], |E_c| is the number of the adjacent pairs of class c.
        (class index, not whole graph index)
        Ak_c: the adjacency matrix of a class, which is the submatrix of 
        the adjacency matrix of the whole graph A'''
        A1=torch.where(torch.logical_and(Ak_c<torch.inf, Ak_c>0),1,0)
        edge_index_c, edge_weight_c=torch_geometric.utils.dense_to_sparse(A1)
        return edge_index_c
    
    def adjacent_diff_env_edge_index(self, Ak_c, env_id_c):
        '''return the edge index of the node pairs within k hops of class c, and they must come from different envs 
        dim=[2, |E_c|], |E_c| is the number of the adjacent pairs of class c.
        (class index, not whole graph index)
        Ak_c: the adjacency matrix of a class, which is the submatrix of 
        the adjacency matrix of the whole graph A'''
        edge_index_c, edge_weight_c=torch_geometric.utils.dense_to_sparse(torch.where(torch.logical_and(Ak_c<torch.inf, Ak_c>0),1,0))
        return edge_index_c[:,torch.where(env_id_c[edge_index_c[0]]!=env_id_c[edge_index_c[1]])[0]]


    def adjacent_rep_distance(self,edge_index_c_1, edge_index_c_2,rep_c):
        '''return the distances of all pairs within k hops of class c, 
        dim=[|E_c|]
        edge_index_c: edge index of the node pairs within k hops of class c 
        (class index, not whole graph index)
        rep_c: rep of class c, extracted from whole graph rep'''
        return torch.cdist(rep_c[edge_index_c_1].unsqueeze(1), rep_c[edge_index_c_2].unsqueeze(1)).squeeze(1).squeeze(1)


    def num_neighbour_diff_same_class(self, Ak, edge_index_c,  c, neighbor_label_counts_c):
        '''return:  
        num_diff_diff: differences of the numbers of different classes between two adjacent nodes of a class c_cur
        dim=[|E_c|]
        num_diff_same: differences of the numbers of same classes between two adjacent nodes of a class c_cur
        dim=[|E_c|]

        parameter:
        c_cur: current class
        '''
        E_c=edge_index_c.shape[1]
        num_diff_diff=torch.zeros(E_c, device=self.config.device) #[|E_c|]
        src_label_counts = neighbor_label_counts_c[edge_index_c[0]]
        tgt_label_counts = neighbor_label_counts_c[edge_index_c[1]]
        num_diff_diff = torch.sum(torch.abs(src_label_counts - tgt_label_counts), dim=1)
        num_diff_same = torch.abs(src_label_counts[:,c] - tgt_label_counts[:,c])
        one_float=torch.tensor(1, device=self.config.device).float()
        return num_diff_diff, torch.where(num_diff_same==0,one_float,num_diff_same)

    def adjacent_hops(self, Ak_c, edge_index_c):
        '''return the minimal hops. dim=[|E_c|]'''
        node1s, node2s=edge_index_c[0], edge_index_c[1]
        return Ak_c[node1s, node2s]

    def cal_CIA_reweighting_loss(self, A, Y, rep, env_id=None):
        '''
        compute CIA loss with reweighting
        if env_id!=None, then use ground truth env label, and only condsider pairs 
        with different env labels'''
        if Y.dim()>1:
            Y.squeeze_(1)
        Y=Y.long()
        hops= self.config.ood.extra_param[13] if env_id is not None else self.config.ood.extra_param[8]
        # calculate the nearst path matrix
        Ak=self.cal_Ak(A, hops)
        num_classes=int(torch.max(Y))+1
        labels_one_hot = torch.nn.functional.one_hot(Y, num_classes)
        # Sum the one-hot encodings along the neighbor dimension to get a count of each label for the neighbors of each node
        neighbor_label_counts = torch.matmul(A, labels_one_hot.float()) # dim=[N,C], for computing num_diff_diff
        del labels_one_hot # save memory

        CIA_rwt_loss=0.

        for c in range(num_classes):
            Ak_c=self.get_Ak_c(Ak, c, Y)
            y_id=torch.where(Y==c)[0]
            rep_c=rep[y_id]
            edge_index_c=self.adjacent_edge_index(Ak_c)
            if edge_index_c.numel()==0:
                continue
            neighbor_label_counts_c=neighbor_label_counts[y_id]
            # compute differences in the number of homophilous/heterophilous neighbors
            num_diff_diff, num_diff_same=self.num_neighbour_diff_same_class(Ak, edge_index_c, c, neighbor_label_counts_c)
            loss_weight=torch.softmax((num_diff_diff/(self.adjacent_hops(Ak_c,edge_index_c)*num_diff_same)), dim=-1)
            loss_weight=(loss_weight-loss_weight.min())/(loss_weight.max()-loss_weight.min())
            adj_loss=self.adjacent_rep_distance(edge_index_c_1=edge_index_c[0], \
                                                edge_index_c_2=edge_index_c[1],rep_c=rep_c)
            CIA_rwt_loss=CIA_rwt_loss + torch.mean(adj_loss*loss_weight)
        return CIA_rwt_loss

    def CIA_loss(self, gt,env_id, num_envs, device):
        '''calculate the CIA loss of a kind of environment partition'''
        all_distances=[]
        #print(F'#in# int(torch.max(gt))+1={int(torch.max(gt))+1}, num_envs={num_envs}')
        for c in range(int(torch.max(gt))+1):
            class_mask=(gt == c)#dimension:[num_samples], composed by `True` and `False`, samples from e1 is set to be True
            for e1 in range(num_envs):
                e1_mask=(env_id == e1)#dimension:[num_samples], composed by `True` and `False`, samples from e2 is set to be True
                class_index_c_e1=torch.where(torch.logical_and(class_mask, e1_mask))[0] # index of samples from class c env e1
                for e2 in range(e1+1, num_envs):
                    # Find indices of samples of class c in environment e
                    #print(F'#in#  {c},{e1},{e2}')

                    e2_mask=(env_id == e2)#dimension:[num_samples], composed by `True` and `False`, samples from c is set to be True
                    class_index_c_e2=torch.where(torch.logical_and(class_mask, e2_mask))[0] # index of samples from class c env e2
                    #print(F'#in# class_index_c_e {class_index_c_e1} {class_index_c_e2}')
                    if class_index_c_e1.numel()==0 or class_index_c_e2.numel()==0: # no samples satisfied
                        continue
                    else:
                        distance = torch.cdist(self.rep[class_index_c_e1], self.rep[class_index_c_e2])
                    #print(F'#in# distance {distance}')
                    all_distances.append(distance)
                    #print(F'#in# dis={distance.shape}') # 2-dim tensor
        # Make sure to handle the case where there are no distances to avoid a crash
        if all_distances:
            CIA_loss = torch.mean(torch.cat([d.mean(dim=-1) for d in all_distances]))
        else:
            CIA_loss = torch.tensor(0.0, device=device)  # Adjust as necessary
        return CIA_loss



