r"""
The implementation of `Handling Distribution Shifts on Graphs: An Invariance Perspective <https://arxiv.org/abs/2202.02466>`_.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph

from GOOD import register
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GATs import GATFeatExtractor



@register.model_register
class E_ER_MGAT(GNNBasic):
    r"""
    E_ER_M implementation adapted from https://github.com/qitianwu/GraphOOD-E_ER_M.
    """
    def __init__(self, config):
        super(E_ER_MGAT, self).__init__(config)
        self.gnn = GATFeatExtractor(config)
        self.p = 0.2
        self.K = config.ood.extra_param[0]
        self.T = config.ood.extra_param[1]
        self.num_sample = config.ood.extra_param[2] # how many nodes for a node should be motified the link with
        self.classifier = Classifier(config)

        #self.gl = Graph_Editer(self.K, config.dataset.num_train_nodes, config.device) #config.dataset.num_train_nodes=420
        self.gl = Graph_Editer(self.K, config.dataset.num_train_nodes, config.device)
        #config.dataset.num_train_nodes is set in GOOD/GOOD/data/good_datasets/good_cbas.py
        self.gl.reset_parameters()
        self.gl_optimizer = torch.optim.Adam(self.gl.parameters(), lr=config.ood.extra_param[3])

    def reset_parameters(self):
        self.gnn.reset_parameters()
        if hasattr(self, 'graph_est'):
            self.gl.reset_parameters()

    def forward(self, *args, **kwargs):
        data = kwargs.get('data')
        loss_func = self.config.metric.loss_func

        # --- K fold ---
        if self.training:
            edge_index, _ = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)
            #print(F'#in#data.x.shape={data.x.shape}') #[689,4]
            #print(F'#in#data.train_mask.shape={data.train_mask.shape}') #[689]
            #print(F'#in#data.train_mask={data.train_mask}')
            #print(F'#in#edge_index max={torch.max(edge_index)}') #413
            x = data.x[data.train_mask]
            y = data.y[data.train_mask]
            #print(F'#in# 3333 x.shape={x.shape}') #[414]
            # --- check will orig_edge_index change? ---
            orig_edge_index = edge_index
            for t in range(self.T):
                Loss, Log_p = [], 0
                env_list=[]
                for k in range(self.K):
                    edge_index, log_p = self.gl(orig_edge_index, self.num_sample, k)
                    #print(F'#in#edge_index max2={torch.max(edge_index)}') #413
                    #print(F'#in#x={ x }')
                    #print(F'#in#edge_index={ edge_index }')
                    #print(F'#in#y={ y }')
                    #torch.save(edge_index, '/data1/qxwang/codes/za/edge_index.pt')
                    rep=self.gnn(data=Data(x=x, edge_index=edge_index, y=y))
                    if t==self.T-1:
                        env_list.append({"edge_index":edge_index,"y":y,"rep":rep})
                    raw_pred = self.classifier(rep)
                    loss = loss_func(raw_pred, y)
                    Loss.append(loss.view(-1))
                    Log_p += log_p
                
                Var, Mean = torch.var_mean(torch.cat(Loss, dim=0))
                reward = Var.detach()
                inner_loss = - reward * Log_p
                self.gl_optimizer.zero_grad()
                inner_loss.backward()
                self.gl_optimizer.step()
            
            CIA_loss=0.
            #print(F'#in# int(torch.max(gt))+1={int(torch.max(gt))+1}, num_envs={num_envs}')
            #print(env_id)
            norm_cnt=0
            num_classes=self.config.dataset.num_classes
            num_envs=self.K
            for c in range(num_classes):
                loss_c=0.
                for e1 in range(num_envs):
                    y1=env_list[e1]["y"]
                    class_mask1=(y1 == c)#dimension:[num_samples], composed by `True` and `False`, samples from e1 is set to be True
                    class_index_c_e1=torch.where(class_mask1)[0] # index of samples from class c env e1
                    rep1=env_list[e1]["rep"]
                    for e2 in range(e1+1, num_envs):
                        # Find indices of samples of class c in environment e
                        y2=env_list[e2]["y"]
                        class_mask2=(y2 == c)
                        class_index_c_e2=torch.where(class_mask2)[0] # index of samples from class c env e2
                        rep2=env_list[e2]["rep"]
                        #print(F'#in# class_index_c_e {class_index_c_e1} {class_index_c_e2}')
                        if class_index_c_e1.numel()==0 or class_index_c_e2.numel()==0: # no samples satisfied
                            continue
                        else:
                            distance = torch.cdist(rep1[class_index_c_e1], rep2[class_index_c_e2]) # [num_c_e1, num_c_e2]
                        #print(F'#in# distance {distance}')
                        #all_distances.append(distance) # ori
                        loss_c+=torch.sum(distance) # for NC
                        norm_cnt+=class_index_c_e1.shape[0]*class_index_c_e2.shape[0]
                #print(f'#in#',loss_c)
                CIA_loss+=loss_c
                        

                #print(f'#in#{torch.sum(torch.tensor(all_distances))}')
            CIA_loss=CIA_loss/norm_cnt

            return CIA_loss, Mean
        else:
            out = self.classifier(self.gnn(data=data))
            return out


class Graph_Editer(nn.Module):
    r"""
    E_ER_M's graph editer adapted from https://github.com/qitianwu/GraphOOD-E_ER_M.
    """
    def __init__(self, K, n, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(K, n, n)) # this is the environment generator
        self.n = n
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, num_sample, k):
        n = self.n
        #print(F'#in#self.n={self.n}')
        Bk = self.B[k]
        
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(torch.int)
        #print(F'#in#A.shape={A.shape}')
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float).to(self.device) # [n, n]
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1. # randomly pick s nodes for each node i=1,2,...,n , 
        # and then reverse the link between node i and its corresponding selected nodes. 
        # (if there is a link between i and j, delete it; if not, add it).
        C = A + M * (A_c - A)
        #print(F'#in#C.shape={C.shape}')
        edge_index = dense_to_sparse(C)[0]

        log_p = torch.sum(
            torch.sum(Bk[S, col_idx], dim=1) - torch.logsumexp(Bk, dim=0)
        )

        return edge_index, log_p
