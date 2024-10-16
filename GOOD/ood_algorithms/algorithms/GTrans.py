"""
Implementation of the baseline GTrans
"""
import torch
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from torch.nn.parameter import Parameter
import os
from typing import Tuple
from tqdm import tqdm
import random
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj, is_undirected, to_undirected
import torch_sparse
from torch_sparse import coalesce


@register.ood_alg_register
class GTrans(BaseOODAlg):
    r"""
    Implementation of the baseline GTrans

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GTrans, self).__init__(config)
        self.n=None
        self.eps = 1e-7
        self.make_undirected=True
        
    '''def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        pred, self.rep=model_output
        return pred'''

    def augment(self, model, strategy='dropedge', p=0.5, edge_index=None, edge_weight=None):
        if hasattr(self, 'delta_feat'):
            delta_feat = self.delta_feat
            #print(f'#in#delta_feat 1 {delta_feat.requires_grad}')
            #print(f'#in#self.feat {self.feat.requires_grad}')
            feat = self.feat + delta_feat
            #print(f'#in# delta_feat 2 {delta_feat.requires_grad}')
            #print(f'#in# feat {feat.requires_grad}')
        else:
            feat = self.feat
        if strategy == 'shuffle':
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            #print(f'#in#edge_weight.shape={edge_weight.shape}')
            #print(f'#in#shuf_fts={shuf_fts}')
            output = model.get_embed(shuf_fts, edge_index, edge_weight)
        if strategy == "dropedge":
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropnode":
            feat = self.feat + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropmix":
            feat = self.feat + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)

        if strategy == "dropfeat":
            feat = F.dropout(self.feat, p=p) + self.delta_feat
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "featnoise":
            mean, std = 0, p
            noise = torch.randn(feat.size()) * std + mean
            feat = feat + noise.to(feat.device)
            output = model.get_embed(feat, edge_index)
        #print(f'#in# {strategy} output={output}')
        return output

    def test_time_loss(self, model, feat, edge_index, edge_weight=None, mode='train'):
        #args = self.args
        loss = 0
        # label constitency
        '''if mode == 'eval': # random seed setting
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)'''
        #if args.strategy == 'dropedge':
            # output1 = self.augment(strategy=args.strategy, p=0.5, edge_index=edge_index, edge_weight=edge_weight)
        output1 = self.augment(model, strategy='dropedge', p=0.05, edge_index=edge_index, edge_weight=edge_weight) #TODO
        '''if args.strategy == 'dropnode':
            output1 = self.augment(strategy=args.strategy, p=0.05, edge_index=edge_index, edge_weight=edge_weight)
        if args.strategy == 'rwsample':
            output1 = self.augment(strategy=args.strategy, edge_index=edge_index, edge_weight=edge_weight)'''
        output2 = self.augment(model, strategy='dropedge', p=0.0, edge_index=edge_index, edge_weight=edge_weight)
        output3 = self.augment(model, strategy='shuffle', edge_index=edge_index, edge_weight=edge_weight)

        '''if args.margin != -1:
            loss = inner(output1, output2) - inner_margin(output2, output3, margin=args.margin)
        else:'''
        loss = inner(output1, output2) - inner(output2, output3)

        return loss

    #@torch.enable_grad()
    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ):
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
        if not training:
            with torch.enable_grad():
                x=data.x
                nnodes = x.shape[0]
                self.n=nnodes
                d = x.shape[1]

                self.max_final_samples = 5

                delta_feat = Parameter(torch.FloatTensor(nnodes, d).to(self.config.device))
                self.delta_feat = delta_feat
                delta_feat.data.fill_(1e-7)
                self.optimizer_feat = torch.optim.Adam([delta_feat], lr=self.config.ood.extra_param[0])

                model = kwargs["model"]

                for param in model.parameters():
                    param.requires_grad = False
                model.eval() # should set to eval

                feat, labels = x.to(self.config.device), data.y.to(self.config.device)#.squeeze()
                edge_index = data.edge_index.to(self.config.device)
                self.edge_index, self.feat, self.labels = edge_index, feat, labels
                self.edge_weight = torch.ones(self.edge_index.shape[1]).to(self.config.device)

                n_perturbations = int(0.1 * self.edge_index.shape[1] //2)
                self.sample_random_block(n_perturbations)

                self.perturbed_edge_weight.requires_grad = True
                self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=self.config.ood.extra_param[1])
                edge_index, edge_weight = edge_index, None


                for loop_feat in range(self.config.ood.extra_param[2]):
                    self.optimizer_feat.zero_grad()
                    
                    loss = self.test_time_loss(model, feat+delta_feat, edge_index, edge_weight)
                    loss.backward()
                    self.optimizer_feat.step()

                new_feat = (feat+delta_feat).detach()
                data.x=new_feat

                for loop_adj in range(self.config.ood.extra_param[2]):
                    self.perturbed_edge_weight.requires_grad = True
                    edge_index, edge_weight  = self.get_modified_adj()
                    #print(f'#in# edge_weight {edge_weight.shape}')
                    #print(f'#in# new_feat {new_feat.shape}')
                    #print(f'#in# edge_index {edge_index.shape}')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    loss = self.test_time_loss(model, new_feat, edge_index, edge_weight)

                    gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]
    

                    with torch.no_grad():
                        self.update_edge_weights(n_perturbations, None, gradient)
                        self.perturbed_edge_weight = self.project(
                            n_perturbations, self.perturbed_edge_weight, self.eps)
                        del edge_index, edge_weight #, logits

                    #if it < self.epochs_resampling - 1:
                    self.perturbed_edge_weight.requires_grad = True
                    self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=self.config.ood.extra_param[1])


                if self.config.ood.extra_param[2] != 0:
                    edge_index, edge_weight  = self.get_modified_adj()
                    edge_weight = edge_weight.detach()


                if self.config.ood.extra_param[2] != 0:
                    edge_index, edge_weight = self.sample_final_edges(model, n_perturbations, data)

                model.train()
                for param in model.parameters():
                    param.requires_grad = True
        return data, targets, mask, node_norm

    def project(self, n_perturbations, values, eps, inplace=False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values

    
    def sample_final_edges(self, model, n_perturbations, data):
        best_loss = float('Inf')
        perturbed_edge_weight = self.perturbed_edge_weight.detach()
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0
        feat, labels = data.x.to(self.config.device), data.y.to(self.config.device).squeeze()
        for i in range(self.max_final_samples):
            if best_loss == float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[torch.topk(perturbed_edge_weight, n_perturbations).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
            self.perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_adj()
            with torch.no_grad():
                loss = self.test_time_loss(model, feat, edge_index, edge_weight, mode='eval')
            # Save best sample
            if best_loss > loss:
                best_loss = loss
                print('best_loss:', best_loss)
                best_edges = self.perturbed_edge_weight.clone().cpu()

        # Recover best sample
        self.perturbed_edge_weight.data.copy_(best_edges.to(self.config.device))
        edge_index, edge_weight = self.get_modified_adj()
        edge_mask = edge_weight == 1

        allowed_perturbations = 3 * n_perturbations if self.make_undirected else n_perturbations # 2 * n_perturbations if self.make_undirected else n_perturbations
        edges_after_attack = edge_mask.sum()
        clean_edges = self.edge_index.shape[1]
        assert (edges_after_attack >= clean_edges - allowed_perturbations
                and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
        return edge_index[:, edge_mask], edge_weight[edge_mask]
    
    def sample_random_block(self, n_perturbations):

        
        edge_index = self.edge_index.clone()
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        row, col = edge_index[0], edge_index[1]
        edge_index_id = (2*self.n - row-1)*row//2 + col - row -1 # // is important to get the correct result
        edge_index_id = edge_index_id.long()
        self.current_search_space = edge_index_id
        self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
        self.perturbed_edge_weight = torch.full_like(
            self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
        )
        
        return
        '''for _ in range(self.max_final_samples):

            self.current_search_space = torch.randint(
                self.n_possible_edges, (self.search_space_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.make_undirected:
                self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = linear_to_full_idx(self.n, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return'''
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def get_modified_adj(self):
        if self.make_undirected:
            modified_edge_index, modified_edge_weight = to_symmetric(
                self.modified_edge_index, self.perturbed_edge_weight, self.n
            )
        else:
            modified_edge_index, modified_edge_weight = self.modified_edge_index, self.perturbed_edge_weight
        edge_index = torch.cat((self.edge_index.to(self.config.device), modified_edge_index), dim=-1)
        edge_weight = torch.cat((self.edge_weight.to(self.config.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
        return edge_index, edge_weight

    '''def _update_edge_weights(self, n_perturbations, epoch, gradient):
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        self.perturbed_edge_weight.data.add_(-lr * gradient)
        # We require for technical reasons that all edges in the block have at least a small positive value
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps'''

    def update_edge_weights(self, n_perturbations, epoch, gradient):
        self.optimizer_adj.zero_grad()
        self.perturbed_edge_weight.grad = gradient
        self.optimizer_adj.step()
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        # Add this at the end of the function before returning the loss
        '''if config.train.epoch<40 and config.train.batch_id>config.train.num_batches-2 and config.train.epoch%3==0:
            save_path = '/data1/qxwang/codes/GOOD/visualization/GTrans_t-SNE'
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'GTrans_{config.dataset.dataset_name}_tsne_epoch_{config.train.epoch}.png')
            self.visualize_tsne(self.rep, data.y, data.env_id, file_path, drop_classes=True, num_classes_to_visualize=8)'''

        self.mean_loss = loss.sum() / mask.sum()

        return self.mean_loss



def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def inner_margin(t1, t2, margin):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return F.relu(1-(t1 * t2).sum(1)-margin).mean()


def grad_with_checkpoint(outputs, inputs):
    #print(f'#in# enter')
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        #print(f'#in# {input.requires_grad}')
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs


def to_symmetric(edge_index, edge_weight, n, op='mean'):
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight


def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (
        n
        - 2
        - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    ).long()
    col_idx = (
        lin_idx
        + row_idx
        + 1 - n * (n - 1) // 2
        + (n - row_idx) * ((n - row_idx) - 1) // 2
    )
    return torch.stack((row_idx, col_idx))


def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if (func(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(miu) * func(a) < 0):
            b = miu
        else:
            a = miu
        if ((b - a) <= epsilon):
            break
    return miu