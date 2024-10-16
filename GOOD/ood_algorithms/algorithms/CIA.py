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
class CIA(BaseOODAlg):
    r"""
    CIA: Cross-domain Intra-class Alignment (version 1: use off-the-shelf env partition)

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(CIA, self).__init__(config)
        self.rep=None
        self.saved_rep=None # for MatchDG contras loss
        self.nearest_index={}
        self.num_to_select_dict={}

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
        t31=time.time()
        #A1=torch.where(Ak<torch.inf,1,0)

        E_c=edge_index_c.shape[1]
        t311=time.time()
        #print(f"#IN#t31-t311:{t311-t31}")
        #node1s, node2s=edge_index_c[0], edge_index_c[1]

        #print(f"#in# Memory 6.0: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
        num_diff_diff=torch.zeros(E_c, device=self.config.device) #[|E_c|]
        #num_diff_same=torch.zeros(E_c, device=self.config.device)
        t312=time.time()
        #print(f"#IN#t312-t311:{t312-t311}")
        #print(f"#in# Memory 6.1: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
        #Use the edge index tensor to get the label counts for the nodes specified by edge_index[0] and edge_index[1]
        src_label_counts = neighbor_label_counts_c[edge_index_c[0]]
        tgt_label_counts = neighbor_label_counts_c[edge_index_c[1]]

        
        t34=time.time()
        #print(f"#IN#t34-t31:{t34-t312}")
        #Compute the absolute difference between these two sets of label counts to compute the diff tensor
        num_diff_diff = torch.sum(torch.abs(src_label_counts - tgt_label_counts), dim=1)
        num_diff_same = torch.abs(src_label_counts[:,c] - tgt_label_counts[:,c])

        #print(num_diff_same)
        #print(f"#in# Memory 6.2: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
        ################del src_label_counts, tgt_label_counts

        

        '''num_belong_to_class_y_cur=y_cur_id.numel()
        t343=time.time()


        #Asame_class=torch.zeros(N,N).to(self.config.device)#slow!!!!!!!!!!!!!!!!!!!!!!!!!!!
        t35=time.time()

        #print(f"#in# Memory 6.3: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
        Asame_class.zero_()
        t351=time.time()

        Asame_class[torch.arange(N).unsqueeze(1).repeat(1,num_belong_to_class_y_cur),y_cur_id.unsqueeze(0).repeat(N,1)]=1#slow!!!!!
        
        t36=time.time()
        print(f"#IN#t36-t351:{t36-t351}")
        ################del y_cur_id, num_belong_to_class_y_cur
        Ac=Asame_class*A1
        #print(f"#in# Memory 6.4: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
        #print(f"#in# node1s {node1s.shape}")

        sum_Ac=torch.sum(Ac, dim=1)
        num_diff_same=torch.abs(sum_Ac[node1s]-sum_Ac[node2s])
        
        #print(f"#in# node1s {node1s.shape}")
        #print(f"#in# Memory 6.5: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
        #print(F'#in# num_diff_same {num_diff_same}')
        #print(f"#in# num_diff_same {num_diff_same.shape}")
        #print(f"#in# Memory 6.6: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
        '''

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
        t01=time.time()

        hops= self.config.ood.extra_param[13] if env_id is not None else self.config.ood.extra_param[8]
        Ak=self.cal_Ak(A, hops)

        t02=time.time()

        num_classes=int(torch.max(Y))+1
        labels_one_hot = torch.nn.functional.one_hot(Y, num_classes)
        # Sum the one-hot encodings along the neighbor dimension 
        # to get a count of each label for the neighbors of each node
        neighbor_label_counts = torch.matmul(A, labels_one_hot.float()) # dim=[N,C], for computing num_diff_diff
        del labels_one_hot # save memory

        CIA_rwt_loss=0.
        t0=time.time()

        N=Ak.shape[0]
        #Asame_class=torch.zeros(N,N, device=self.config.device) # for adjacent_diff_env_edge_index, move it out and initialize it here rather than in 
        # adjacent_diff_env_edge_index for efficiency
        for c in range(num_classes):
            #print(f'#in# c {c }')
            #print(f'#in# Memory 0:{torch.cuda.is_available()}')
            #print(f"#in# Memory 1: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
            t1=time.time()
            
            Ak_c=self.get_Ak_c(Ak, c, Y)
            #print(f"#in# Memory 2: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
            #print(f'#in Ak_c {Ak_c}')
            y_id=torch.where(Y==c)[0]
            #print(f"#in# Memory 3: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
            rep_c=rep[y_id]
            #print(f"#in# Memory 4: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
            #print(f'#in# rep_c {rep_c.shape }')
            if env_id is not None:
                edge_index_c=self.adjacent_diff_env_edge_index(Ak_c, env_id[y_id])
                #print(f"#in# Memory 5.1: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
                #print(f'#in# edge_index_c {edge_index_c.shape}')
            else:
                edge_index_c=self.adjacent_edge_index(Ak_c)
                #print(f"#in# Memory 5.2: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
            if edge_index_c.numel()==0:
                continue
            
            
            t2=time.time()

            #A_c_cur=self.get_Ak_c(A,c,Y) # function get_Ak_c can be used for extracting submatrix of any matrix
            neighbor_label_counts_c=neighbor_label_counts[y_id]
            
            #print(f'#in# Ak {Ak}')
            t3=time.time()
            #print(f"#IN#t3-t2:{t3-t2}")
            num_diff_diff, num_diff_same=self.num_neighbour_diff_same_class(Ak, edge_index_c, c, neighbor_label_counts_c)
            # 3 lines below went well, no bugs
            #print(f'#in# num_diff_diff {num_diff_diff}')
            #print(f'#in# num_diff_same {num_diff_same}')
            #print(f'#in# torch.max(torch.softmax(num_diff_diff/(self.adjacent_hops(Ak_c,edge_index_c)*num_diff_same), dim=-1)) {torch.max(torch.softmax(num_diff_diff/(self.adjacent_hops(Ak_c,edge_index_c)*num_diff_same), dim=-1))}')
            
            t4=time.time()

            if self.config.ood.extra_param[15]==0:
                CIA_rwt_loss=CIA_rwt_loss + torch.mean(self.adjacent_rep_distance(edge_index_c[0],edge_index_c[1], rep_c)*\
                                                    torch.softmax(num_diff_diff/(self.adjacent_hops(Ak_c,edge_index_c)*num_diff_same), dim=-1), dim=-1)# good empirical performance on cora
            else:
                loss_weight=torch.softmax((num_diff_diff/(self.adjacent_hops(Ak_c,edge_index_c)*num_diff_same)), dim=-1)
                loss_weight=(loss_weight-loss_weight.min())/(loss_weight.max()-loss_weight.min())
                
                loss_weight_larger_than_threshold_id=torch.where(loss_weight>0.1)[0]
                if loss_weight_larger_than_threshold_id.numel()==0:
                    continue
                
                adj_loss=self.adjacent_rep_distance(edge_index_c_1=edge_index_c[0][loss_weight_larger_than_threshold_id], \
                                                    edge_index_c_2=edge_index_c[1][loss_weight_larger_than_threshold_id],rep_c=rep_c)
                CIA_rwt_loss=CIA_rwt_loss + torch.mean(adj_loss*loss_weight[loss_weight_larger_than_threshold_id])
            #print(f'#in# adj_loss {adj_loss}')
            #print(f'#in# loss_weight_larger_than_threshold_id {loss_weight[loss_weight_larger_than_threshold_id]}')
           
            t5=time.time()
        
        t_end=time.time()
        #print(f"#IN#t_end-t0:{t_end-t0}")
        return CIA_rwt_loss

    def ratio_neighbour_diff_same_class(self, Ak, edge_index_c,  c, neighbor_label_ratio_c):
        '''return:  
        num_diff_diff: differences of the numbers of different classes between two adjacent nodes of a class c_cur
        dim=[|E_c|]
        num_diff_same: differences of the numbers of same classes between two adjacent nodes of a class c_cur
        dim=[|E_c|]

        parameter:
        c_cur: current class
        '''
        t31=time.time()
        #A1=torch.where(Ak<torch.inf,1,0)

        E_c=edge_index_c.shape[1]
        t311=time.time()
        #print(f"#IN#t31-t311:{t311-t31}")
        #node1s, node2s=edge_index_c[0], edge_index_c[1]

        #print(f"#in# Memory 6.0: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
        num_diff_diff=torch.zeros(E_c, device=self.config.device) #[|E_c|]
        #num_diff_same=torch.zeros(E_c, device=self.config.device)
        t312=time.time()
        #print(f"#IN#t312-t311:{t312-t311}")
        #print(f"#in# Memory 6.1: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
        #Use the edge index tensor to get the label counts for the nodes specified by edge_index[0] and edge_index[1]
        src_label_counts = neighbor_label_ratio_c[edge_index_c[0]]
        tgt_label_counts = neighbor_label_ratio_c[edge_index_c[1]]

        
        t34=time.time()
        #print(f"#IN#t34-t31:{t34-t312}")
        #Compute the absolute difference between these two sets of label counts to compute the diff tensor
        num_diff_diff = torch.sum(torch.abs(src_label_counts - tgt_label_counts), dim=1)
        num_diff_same = torch.abs(src_label_counts[:,c] - tgt_label_counts[:,c])
        one_float=torch.tensor(1, device=self.config.device).float()
        return num_diff_diff, torch.where(num_diff_same==0,one_float,num_diff_same)


    def CIA_LRA_loss(self, A, Y, rep, env_id=None):
        '''
        compute CIA loss with reweighting
        if env_id!=None, then use ground truth env label, and only condsider pairs 
        with different env labels
        
        use nerighboring ratio rather than number,
        add pair-wise normalization, for ICML exp 
        '''
        if Y.dim()>1:
            Y.squeeze_(1)
        Y=Y.long()
        t01=time.time()

        hops= self.config.ood.extra_param[13] if env_id is not None else self.config.ood.extra_param[8]
        Ak=self.cal_Ak(A, hops)

        t02=time.time()

        num_classes=int(torch.max(Y))+1
        labels_one_hot = torch.nn.functional.one_hot(Y, num_classes)
        # Sum the one-hot encodings along the neighbor dimension 
        # to get a count of each label for the neighbors of each node
        N=Ak.shape[0]
        #print("#in# ori A",A)
        A[range(N),range(N)]=1
        normalized_A=A/torch.sum(A,dim=1).unsqueeze(1).repeat(1,A.shape[0])
        #print("#in# A",normalized_A)
        #print('#in# torch.sum(A,dim=1).unsqueeze(1).repeat(1,A.shape[0])',torch.sum(A,dim=1).unsqueeze(1).repeat(1,A.shape[0]))
        neighbor_label_ratio=labels_one_hot.float()
        for i in range(self.config.model.model_layer):
            neighbor_label_ratio = torch.matmul(normalized_A, neighbor_label_ratio) # dim=[N,C], for computing num_diff_diff
            #print(f"#in# i={i} neighbor_label_ratio",neighbor_label_ratio)
        del labels_one_hot # save memory

        CIA_rwt_loss=0.
        t0=time.time()

        
        #Asame_class=torch.zeros(N,N, device=self.config.device) # for adjacent_diff_env_edge_index, move it out and initialize it here rather than in 
        # adjacent_diff_env_edge_index for efficiency
        norm_cnt=0

        for c in range(num_classes):
            #print(f'#in# c {c }')
            #print(f'#in# Memory 0:{torch.cuda.is_available()}')
            #print(f"#in# Memory 1: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
            t1=time.time()
            
            Ak_c=self.get_Ak_c(Ak, c, Y)
            #print(f"#in# Memory 2: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
            #print(f'#in Ak_c {Ak_c}')
            y_id=torch.where(Y==c)[0]
            #print(f"#in# Memory 3: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
            rep_c=rep[y_id]
            #print(f"#in# Memory 4: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
            #print(f'#in# rep_c {rep_c.shape }')
            if env_id is not None:
                edge_index_c=self.adjacent_diff_env_edge_index(Ak_c, env_id[y_id])
                #print(f"#in# Memory 5.1: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
                #print(f'#in# edge_index_c {edge_index_c.shape}')
            else:
                edge_index_c=self.adjacent_edge_index(Ak_c)
                #print(f"#in# Memory 5.2: {torch.cuda.memory_allocated( torch.device('cuda:0')) / 1024 ** 2} MB")
            if edge_index_c.numel()==0:
                continue
            
            
            t2=time.time()

            #A_c_cur=self.get_Ak_c(A,c,Y) # function get_Ak_c can be used for extracting submatrix of any matrix
            neighbor_label_ratio_c=neighbor_label_ratio[y_id]
            
            #print(f'#in# Ak {Ak}')
            t3=time.time()
            #print(f"#IN#t3-t2:{t3-t2}")
            #print('#in# edge_index_c', edge_index_c)
            #print('#in# neighbor_label_ratio_c', neighbor_label_ratio_c)
            ratio_diff_diff, ratio_diff_same=self.ratio_neighbour_diff_same_class(Ak, edge_index_c, c, neighbor_label_ratio_c)
            
            if self.config.ood.extra_param[15]==0:
                loss_weight=ratio_diff_diff/(self.adjacent_hops(Ak_c,edge_index_c)*ratio_diff_same)
            elif self.config.ood.extra_param[15]==1:
                loss_weight=1/(self.adjacent_hops(Ak_c,edge_index_c)*ratio_diff_same) # no Q^diff
            elif self.config.ood.extra_param[15]==2:
                loss_weight=ratio_diff_diff/(self.adjacent_hops(Ak_c,edge_index_c)) # no Q^same
            elif self.config.ood.extra_param[15]==3:
                loss_weight=ratio_diff_diff/(ratio_diff_same)# no d(i,j)
            elif self.config.ood.extra_param[15]==4:
                loss_weight=(ratio_diff_diff+ratio_diff_same)/self.adjacent_hops(Ak_c,edge_index_c)
            loss_weight=(loss_weight-loss_weight.min())/(loss_weight.max()-loss_weight.min())
            #print(loss_weight.max(),loss_weight.min())
            #print('#in#',(loss_weight>0.1).sum()/(loss_weight.numel()))
            #print((loss_weight>0.3).sum()/(loss_weight.numel()))
            #print((loss_weight>0.5).sum()/(loss_weight.numel()))
            #print('#in# 111',self.config.dataset.dataset_name)
            #print('#in# 222',torch.where(loss_weight>0.0)[0])
            loss_weight_larger_than_threshold_id=torch.where(loss_weight>0.1)[0]
            norm_cnt+=loss_weight_larger_than_threshold_id.shape[0]
            if loss_weight_larger_than_threshold_id.numel()==0:
                continue
            #loss_weight_larger_than_threshold_mask=torch.where(loss_weight>0.99,1,0)
            
            adj_loss=self.adjacent_rep_distance(edge_index_c_1=edge_index_c[0][loss_weight_larger_than_threshold_id], \
                                                edge_index_c_2=edge_index_c[1][loss_weight_larger_than_threshold_id],rep_c=rep_c)
            #adj_loss=torch.mean((adj_loss.detach()*(1-loss_weight_larger_than_threshold_mask)+(adj_loss-adj_loss.detach())*loss_weight_larger_than_threshold_mask)*loss_weight, dim=-1)
            
            #
            #print(f'#in# {loss_weight_larger_than_threshold_id.shape}')
            
            #grad=torch.autograd.grad(torch.mean(adj_loss), rep_c)[0]
            #print(f'#in#{grad.shape}')
            #CIA_rwt_loss=CIA_rwt_loss + torch.mean(adj_loss*loss_weight[loss_weight_larger_than_threshold_id]) # ori
            #print(f'#in#',torch.sum(adj_loss*loss_weight[loss_weight_larger_than_threshold_id]))
            #print(loss_weight_larger_than_threshold_id.numel())
            CIA_rwt_loss=CIA_rwt_loss + torch.sum(adj_loss*loss_weight[loss_weight_larger_than_threshold_id]) #NC
            
            #print(f'#in# adj_loss {adj_loss}')
            #print(f'#in# loss_weight_larger_than_threshold_id {loss_weight[loss_weight_larger_than_threshold_id]}')
           
            t5=time.time()
        if norm_cnt==0:
            CIA_rwt_loss=0
        else:
            CIA_rwt_loss=CIA_rwt_loss/norm_cnt
        
        t_end=time.time()
        #print(f"#IN#t_end-t0:{t_end-t0}")
        return CIA_rwt_loss


    def CIA_loss_ori(self, gt,env_id, num_envs, device):
        '''calculate the CIA loss of a kind of environment partition
        original ICLR cold, good emprirical performance on GCN'''
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

    def CIA_loss(self, gt,env_id, num_envs, device):
        '''calculate the CIA loss of a kind of environment partition
        add pair-wise normalization, for ICML exp
        '''
        all_distances=[] 
        CIA_loss=0.
        #print(F'#in# int(torch.max(gt))+1={int(torch.max(gt))+1}, num_envs={num_envs}')
        #print(env_id)
        norm_cnt=0
        for c in range(int(torch.max(gt))+1):
            loss_c=0.
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
                        distance = torch.cdist(self.rep[class_index_c_e1], self.rep[class_index_c_e2]) # [num_c_e1, num_c_e2]
                    #print(F'#in# distance {distance}')
                    #all_distances.append(distance) # ori
                    loss_c+=torch.sum(distance) # for NC
                    norm_cnt+=class_index_c_e1.shape[0]*class_index_c_e2.shape[0]
            #print(f'#in#',loss_c)
            CIA_loss+=loss_c
                    

            #print(f'#in#{torch.sum(torch.tensor(all_distances))}')
        CIA_loss=CIA_loss/norm_cnt # for NC exp
        '''# Make sure to handle the case where there are no distances to avoid a crash
        if all_distances:
            #CIA_loss = torch.mean(torch.cat([d.mean(dim=-1) for d in all_distances])) # for all original results
            CIA_loss = torch.sum(torch.cat([d.sum(dim=-1) for d in all_distances])) # for neural collapse exp
            CIA_loss/=(+)
        else:
            CIA_loss = torch.tensor(0.0, device=device)  # Adjust as necessary'''
        return CIA_loss
    

    def cos_sim(self, x,y):
        """
        args:
        x: (N, D)
        y: (M, D)
        returns:
        (N, M), cos sim of vectors in x and in y
        """
        # 标准化向量
        #print(f'#in# x={x} y={y}')
        #print(f'#in# x.norm(dim=1, keepdim=True)={x.norm(dim=1, keepdim=True)}, y.norm(dim=1, keepdim=True)={y.norm(dim=1, keepdim=True)}')
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)
        
        # 计算余弦相似度
        #print(f'#in# x_norm={x_norm}m y_norm={y_norm}')
        res = torch.einsum('nd,md->nm', x_norm, y_norm)/0.1
        #print(f'#in# res={res}')
        return res

    def MatchDG_contras_loss(self, gt, env_id, num_envs, device):
        '''calculate the CIA loss of a kind of environment partition'''
        all_distances=[] 
        total_contras_loss=0.
        #print(F'#in# int(torch.max(gt))+1={int(torch.max(gt))+1}, num_envs={num_envs}')
        #print(env_id)
        norm_cnt=0
        for c in range(int(torch.max(gt))+1):
            contras_loss_c=0.
            #print(f'#in# c={c}')
            class_mask=(gt == c)#dimension:[num_samples], composed by `True` and `False`, samples from e1 is set to be True
            #print(f'#in# -1 gt={gt}')
            for e1 in range(num_envs):
                e1_mask=(env_id == e1)#dimension:[num_samples], composed by `True` and `False`, samples from e2 is set to be True
                class_index_c_e1=torch.where(torch.logical_and(class_mask, e1_mask))[0] # index of samples from class c env e1
                #print(f'#in# 0 gt={gt}')
                for e2 in range(e1+1, num_envs):
                    # Find indices of samples of class c in environment e
                    #print(F'#in#  {c},{e1},{e2}')

                    e2_mask=(env_id == e2)#dimension:[num_samples], composed by `True` and `False`, samples from c is set to be True
                    class_index_c_e2=torch.where(torch.logical_and(class_mask, e2_mask))[0] # index of samples from class c env e2
                    #print(F'#in# class_index_c_e {class_index_c_e1} {class_index_c_e2}')
                    #print(f'#in# 1 gt={gt}')

                    if class_index_c_e1.numel()==0 or class_index_c_e2.numel()==0: # no samples satisfied
                            continue
                            
                    #if self.config.train.epoch%5==0:
                        #print(f'#in# in gt={gt}')
                    # Find the top 20% nearest node points in a class, and take them as the positive pairs
                    # Since there are too many nodes in batch for graphs, we can't iterate on every node to find
                    # its nearst nodes.
                    distance = torch.cdist(self.rep[class_index_c_e1], self.rep[class_index_c_e2]) # [num_c_e1, num_c_e2]
                    num_elements = distance.numel()
                    num_to_select = int(num_elements * 0.2)  # top 20% nearst
                    #self.num_to_select_dict[f'{c}_{e1}_{e2}']=num_to_select
                    flattened_distance = distance.view(-1)
                    values, indices = torch.topk(flattened_distance, num_to_select, largest=False)
                    # mapping the flattened indices to the original indices
                    rows = indices // distance.size(1)
                    cols = indices % distance.size(1)
                    nearest_index_c_e1 = class_index_c_e1[rows]
                    nearest_index_c_e2 = class_index_c_e2[cols]
                    #self.nearest_index[f'{c}_{e1}']=nearest_index_c_e1
                    #self.nearest_index[f'{c}_{e2}']=nearest_index_c_e2

                    #print(f'#in# 1.5 gt={gt}')

                    #if f'{c}_{e1}' not in self.nearest_index.keys() or f'{c}_{e2}' not in self.nearest_index.keys(): # no samples satisfied
                    #        continue

                    #print(f'#in# 2 gt={gt}')
                    #print(f'#in# index',self.nearest_index[f'{c}_{e1}'], self.nearest_index[f'{c}_{e2}'])
                    #print(f'#in# reps', self.rep[self.nearest_index[f'{c}_{e1}']], self.rep[self.nearest_index[f'{c}_{e1}']])
                    pos_dis=self.cos_sim(self.rep[nearest_index_c_e1], self.rep[nearest_index_c_e2]) # [nearest_c_e1, nearest_c_e2]
                    #print(f'#in# pos_dis {pos_dis}')
                    #print(f'#in# 3 gt={gt}')

                    # select the same number of nodes from other classes
                    #num_to_select=self.num_to_select_dict[f'{c}_{e1}_{e2}']

                    #print(f'#in# num_to_select={num_to_select}')
                    #print(f'#in# gt={gt}')
                    #print(f'#in# gt.shape={gt.shape}')
                    #print(f'#in# torch.max(gt)={torch.max(gt)}')
                    
                    #print(f'#in# gt!=c {gt!=c}')
                    other_class_index= torch.where(gt != c)[0]
                    num_to_select = len(other_class_index) if num_to_select>len(other_class_index) else num_to_select
                    rand_index = torch.randperm(len(other_class_index))[:num_to_select]
                    other_class_index = other_class_index[rand_index]
                    neg_dis = self.cos_sim(self.rep[nearest_index_c_e1], self.rep[other_class_index]) # [nearest_c_e1, num_to_select]

                    contras_loss_c += torch.sum( - torch.log(torch.exp(pos_dis)/(torch.exp(pos_dis)+torch.sum(torch.exp(neg_dis), dim=1).unsqueeze(1).repeat(1,pos_dis.shape[1])) ) )

                    norm_cnt+=num_to_select
            ##print(f'#in#',loss_c)
            total_contras_loss+=contras_loss_c
        
        return total_contras_loss/norm_cnt

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
            config = munchify({device: torch.device('cuda'),
                                   dataset: {num_envs: int(10)},
                                   ood: {ood_param: float(0.1)}
                                   })
        Returns (Tensor):
            loss with CIA penalty

        """

        #print(f'#in#666')
        mean_loss = loss.sum() / mask.sum() #vanilla ce loss
        #spec_loss=CIA_reweighting_loss
        loss=mean_loss # for backward

        spec_loss={}
        # gt env partition
        if config.ood.extra_param[10] and config.train.epoch>1:
            CIA_loss_gt_env=self.CIA_loss(data.y, data.env_id, config.dataset.num_envs, loss.device)
            #CIA_loss_gt_env=self.CIA_loss_ori(data.y, data.env_id, config.dataset.num_envs, loss.device)
            spec_loss['CIA_loss_gt_env']=CIA_loss_gt_env
            loss=loss+config.ood.extra_param[0]*CIA_loss_gt_env


        elif config.ood.extra_param[10]==2:
            if self.config.train.epoch<5:
                ##print(f'#in# data.y={data.y}')
                CIA_loss_gt_env=self.MatchDG_contras_loss(data.y, data.env_id, config.dataset.num_envs, loss.device)
                spec_loss['MatchDG_contras_loss']=CIA_loss_gt_env
                loss=loss+config.ood.extra_param[0]*CIA_loss_gt_env
            else:
                CIA_loss_gt_env=self.CIA_loss(data.y, data.env_id, config.dataset.num_envs, loss.device)
                spec_loss['CIA_loss_gt_env']=CIA_loss_gt_env
                loss=loss+config.ood.extra_param[0]*CIA_loss_gt_env
                
                
        # node feature clustering partition
        if config.ood.extra_param[1]:
            K=config.ood.extra_param[4] # num of clusters
            CIA_loss_node_feature_clustering_env=self.CIA_loss(data.y, kwargs['node_feature_cluster_ids'], int(K), loss.device)
            spec_loss['CIA_loss_node_feature_clustering_env']=CIA_loss_node_feature_clustering_env
            loss=loss+config.ood.extra_param[6]*CIA_loss_node_feature_clustering_env

        # local reweighting CIA loss
        if config.ood.extra_param[7] and config.train.epoch>1:
            #print(f'#in# data.edge_index {data.edge_index.shape}')
            A=torch_geometric.utils.to_dense_adj(data.edge_index).squeeze(0)
            #print(f'#in# A {A.shape}')
            
            CIA_reweighting_loss=self.CIA_LRA_loss(A, data.y, rep=self.rep)
            spec_loss['CIA_reweighting_loss']=CIA_reweighting_loss
            loss=loss+config.ood.extra_param[9]*CIA_reweighting_loss

        # gt env CIA reweighting
        if config.ood.extra_param[11]:
            A=torch_geometric.utils.to_dense_adj(data.edge_index).squeeze(0)
            CIA_gt_env_reweighting_loss=self.cal_CIA_reweighting_loss(A, data.y, rep=self.rep, env_id=data.env_id)
            spec_loss['CIA_gt_env_reweighting_loss']=CIA_gt_env_reweighting_loss
            loss=loss+config.ood.extra_param[12]*CIA_gt_env_reweighting_loss

        self.mean_loss = mean_loss
        self.spec_loss = spec_loss

        #print(f'#in#{config.ood.extra_param[14]}')
        if config.ood.extra_param[14]:
            writer=kwargs['writer']
            var, dis=self.var_dis(self.rep, data.y)
            print(f'#in# {var/config.dataset.num_classes} {dis/config.dataset.num_classes**2}')
            return loss, var/config.dataset.num_classes, dis/config.dataset.num_classes**2

        # Add this at the end of the function before returning the loss
        '''if  config.train.batch_id>config.train.num_batches-2 and config.train.epoch%3==0 and config.train.epoch>90:
            #print(F'#in# {K}')
            save_path = '/data1/qxwang/codes/GOOD/visualization/CIA_t-SNE'
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'CIA_reweighting-no-gt-env_{self.config.ood.extra_param[8]}-{config.ood.extra_param[9]}_{config.dataset.dataset_name}_{config.dataset.shift_type}_{config.dataset.domain}_{config.ood.extra_param[9]}_tsne_epoch_{config.train.epoch}.png')
            if int(torch.max(data.y))>8: # if this dataset too many classes
                self.visualize_tsne(self.rep, data.y, data.env_id, file_path, drop_classes=True, num_classes_to_visualize=8)
            else:
                self.visualize_tsne(self.rep, data.y, data.env_id, file_path)'''

        return loss
