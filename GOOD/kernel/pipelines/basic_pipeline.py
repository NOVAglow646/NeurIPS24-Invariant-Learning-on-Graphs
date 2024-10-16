r"""Training pipeline: training/evaluation structure, batch training.
"""
import datetime
import os
import shutil
from typing import Dict
from typing import Union

import numpy as np
import torch
import torch.nn
from munch import Munch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.args import CommonArgs
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.logger import pbar_setting
from GOOD.utils.register import register
from GOOD.utils.train import nan2zero_get_mask

from GOOD.networks.models.CIAGCNs import CIAGCN, CIAGCN_no_center
from kmeans_pytorch import kmeans
from torch.utils.tensorboard import SummaryWriter
import time

from GOOD.networks.model_manager import load_model
import torch.nn.functional as F


@register.pipeline_register
class Pipeline:
    r"""
    Kernel pipeline.

    Args:
        task (str): Current running task. 'train' or 'test'
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    """

    def __init__(self, task: str, model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]],
                 ood_algorithm: BaseOODAlg,
                 config: Union[CommonArgs, Munch]):
        super(Pipeline, self).__init__()
        self.task: str = task
        self.model: torch.nn.Module = model
        self.loader: Union[DataLoader, Dict[str, DataLoader]] = loader
        self.ood_algorithm: BaseOODAlg = ood_algorithm
        self.config: Union[CommonArgs, Munch] = config
        self.batch_cluster_env_id=None
        self.writer=None
        # for NIPS 2024
        self.edge_mask_GNN=load_model(config.model.model_name, config).to(config.device)


    def train_batch(self, data: Batch, pbar, pretrained_model=None, pretrained_inv_model=None) -> dict:
        r"""
        Train a batch. (Project use only)

        Args:
            data (Batch): Current batch of data.

        Returns:
            Calculated loss.
        """


        self.config.num_batches=len(pbar)

        data = data.to(self.config.device)

        self.ood_algorithm.optimizer.zero_grad()

        mask, targets = nan2zero_get_mask(data, 'train', self.config)

        
        node_norm = data.get('node_norm') if self.config.model.model_level == 'node' else None
        #print(f'#in# node_norm {node_norm}')
        #print(f'#in# node_norm.shape {node_norm.shape}')
        if self.config.ood.ood_alg=='sp':
            idx=torch.where(data.env_id>-1)[0]
            node_norm=node_norm[idx]
        node_norm = node_norm.reshape(targets.shape) if node_norm is not None else None
        
        data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                             self.model.training,
                                                                             self.config)
        
        edge_weight = data.get('edge_norm') if self.config.model.model_level == 'node' else None
        #e=data.get('edge_norm') # edge_norm is 1/the prob that a edge is sampled
        #print('#in# data.get(edge_norm)',e.max(), e.min(), torch.mean(e)) # tensor(58.5000, device='cuda:1') tensor(1., device='cuda:1') tensor(5.2483, device='cuda:1')
        #edge_weight=None
        mask_start_epoch=-1
        shift_str=self.config.dataset.shift_type
        domain=self.config.dataset.domain
        dataset_name=self.config.dataset.dataset_name
        #if self.config.dataset.dataset_name=='GOODWebKB':
        #    mask_start_epoch=50
        if self.config.train.epoch>mask_start_epoch and self.config.use_inv_edge_mask and self.config.ood.ood_alg=='CIA':
            if self.config.ood.extra_param[1]==1: # CIA-LRA
                node_features=self.edge_mask_GNN.get_embed(x=data.x, edge_index=data.edge_index, edge_weight=edge_weight)
                if self.config.model.model_name in ['CIAGAT', 'GAT']:
                    edge_weight=compute_edge_mask_sigmoid(node_features=node_features, edges=data.edge_index)
                elif self.config.model.model_name in ['CIAGCN', 'GCN']:
                    #edge_weight=compute_edge_mask_sigmoid(node_features=node_features, edges=data.edge_index)
                    #edge_weight=compute_edge_mask_normalization(node_features=node_features, edges=data.edge_index)
                    #print(f'#in#edge_weight {edge_weight} {edge_weight.max()} {edge_weight.min()} >0:{torch.where(edge_weight>0)[0].numel()/edge_weight.shape[0]} >0.1:{torch.where(edge_weight>0.1)[0].numel()/edge_weight.shape[0]} >0.1:{torch.where(edge_weight>0.1)[0].numel()/edge_weight.shape[0]} >0.5:{torch.where(edge_weight>0.5)[0].numel()/edge_weight.shape[0]} mean:{torch.mean(edge_weight)}')
                    if dataset_name=='GOODCBAS':
                        edge_weight=compute_edge_mask_sigmoid(node_features=node_features, edges=data.edge_index)
                    elif dataset_name=='GOODArxiv' and domain=='time' and shift_str=='concept':
                        edge_weight=compute_edge_mask_sigmoid(node_features=node_features, edges=data.edge_index)
                        edge_weight=edge_weight*data.get('edge_norm')
                    else:
                        edge_weight=compute_edge_mask_normalization(node_features=node_features, edges=data.edge_index)
        
        #print("#in# edge_weight", edge_weight)   
        #print("#in# data", data)    
                
        #print(f'#in# training edge_weight {edge_weight.shape}') # torch.Size([55078])
        model_output = self.model(data=data, edge_weight=edge_weight, ood_algorithm=self.ood_algorithm)
        #print(f'#in#',self.config.ood.ood_alg)
        

        raw_pred = self.ood_algorithm.output_postprocess(model_output, config=self.config)
        if self.config.ood.ood_alg=='sp':
            #print(f"#in# raw_pred={raw_pred}")
            #print(f"#in# targets={targets}")
            raw_pred=raw_pred[idx]
            
            loss = torch.nn.functional.cross_entropy(raw_pred, targets.long(), reduction='none') * mask
        
            #print(F'#in# node_norm={node_norm.shape}') 
            #print(F'#in# mask.sum()={mask.sum()}')
            loss = loss * node_norm * mask.sum() if self.config.model.model_level == 'node' else loss
        else:
            loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config)
        #print(f'#in# {self.ood_algorithm}')

        loss = self.ood_algorithm.loss_postprocess(loss, data, mask, self.config,\
                                                    pretrained_model=pretrained_model, pretrained_inv_model=pretrained_inv_model,\
                                                    edge_weight=edge_weight, ood_algorithm=self.ood_algorithm, writer=self.writer, model=self.model)
        '''if self.config.ood.ood_alg=='GTrans':
            print(f'#in# data={data}')
            print(f'#in# loss={loss}')'''
        


        if self.config.ood.ood_alg!='sp' or self.config.ood.ood_alg=='sp' and self.config.ood.extra_param[0]==0:
            self.ood_algorithm.backward(loss)


        return {'loss': loss.detach()}

    def train(self):
        r"""
        Training pipeline. (Project use only)
        """
        # config model
        print('#D#Config model')
        self.config_model('train')

        # Load training utils
        print('#D#Load training utils')
        self.ood_algorithm.set_up(self.model, self.config, edge_mask_GNN=self.edge_mask_GNN)

        
        # train the model
        time_epochs=[]
            
        
        for epoch in range(self.config.train.ctn_epoch, self.config.train.max_epoch):
            start_time=time.time()
            self.config.train.epoch = epoch
            print(f'#IN#Epoch {epoch}:')

            mean_loss = 0
            spec_loss = 0

            self.ood_algorithm.stage_control(self.config)

            pbar = tqdm(enumerate(self.loader['train']), total=len(self.loader['train']), **pbar_setting)

            for index, data in pbar:
                #print(F'#in# data.batch={data.batch}')

                if data.batch is not None and (data.batch[-1] < self.config.train.train_bs - 1):
                    continue
                #print(F'#in# pbar={len(pbar)}') #10
                # Parameter for DANN
                p = (index / len(self.loader['train']) + epoch) / self.config.train.max_epoch
                self.config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1

                self.config.train.batch_id=index

                # train a batch
                train_stat = self.train_batch(data, pbar)
                mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (index + 1)

                #print(F'#in# 33333')
                if self.ood_algorithm.spec_loss is not None:
                    if isinstance(self.ood_algorithm.spec_loss, dict):
                        desc = f'ML: {mean_loss:.4f}|'
                        for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                            if not isinstance(spec_loss, dict):
                                spec_loss = dict()
                            if loss_name not in spec_loss.keys():
                                spec_loss[loss_name] = 0
                            spec_loss[loss_name] = (spec_loss[loss_name] * index + loss_value) / (index + 1)
                            desc += f'{loss_name}: {spec_loss[loss_name]:.4f}|'
                        pbar.set_description(desc[:-1])
                    else:
                        #print(F'#in# 4444')
                        spec_loss = (spec_loss * index + self.ood_algorithm.spec_loss) / (index + 1)
                        pbar.set_description(f'M/S Loss: {mean_loss:.4f}/{spec_loss:.4f}')
                        #print(F'#in# 5555')
                else:
                    pbar.set_description(f'Loss: {mean_loss:.4f}')

            end_time=time.time()    
            time_cur_epoch=end_time-start_time
            time_epochs.append(time_cur_epoch)
            print(f'#IN#\n Cost {time_cur_epoch} seconds in epoch {epoch}')

            # Eval training score
            
            # Epoch val
            if self.config.ood.ood_alg!='sp' or self.config.ood.ood_alg=='sp' and self.config.ood.extra_param[0]==0:
                print('#IN#\nEvaluating...')
                if self.ood_algorithm.spec_loss is not None:
                    if isinstance(self.ood_algorithm.spec_loss, dict):
                        desc = f'ML: {mean_loss:.4f}|'
                        for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                            desc += f'{loss_name}: {spec_loss[loss_name]:.4f}|'
                        print(f'#IN#Approximated ' + desc[:-1])
                    else:
                        print(f'#IN#Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}')
                else:
                    print(f'#IN#Approximated average training loss {mean_loss.cpu().item():.4f}')


                #print('#IN#\ndebugging...1')
                epoch_train_stat = self.evaluate('eval_train')
                #print('#IN#\ndebugging...2')
                id_val_stat = self.evaluate('id_val')
                #print('#IN#\ndebugging...3')
                id_test_stat = self.evaluate('id_test')
                #print('#IN#\ndebugging...4')
                val_stat = self.evaluate('val')
                #print('#IN#\ndebugging...5')
                test_stat = self.evaluate('test')

                #print('#IN#\ndebugging...6')

                # checkpoints save
                self.save_epoch(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat, self.config)

                # --- scheduler step ---

                self.ood_algorithm.scheduler.step()
        print(f'#IN# Mean runtime (second) per epoch: {np.mean(time_epochs)}')
        print('#IN#Training end.')

    @torch.no_grad()
    def evaluate(self, split: str):
        r"""
        This function is design to collect data results and calculate scores and loss given a dataset subset.
        (For project use only)

        Args:
            split (str): A split string for choosing the corresponding dataloader. Allowed: 'train', 'id_val', 'id_test',
                'val', and 'test'.

        Returns:
            A score and a loss.

        """
        stat = {'score': None, 'loss': None}
        if self.loader.get(split) is None:
            return stat
        self.model.eval()
        self.edge_mask_GNN.eval()

        loss_all = []
        mask_all = []
        pred_all = []
        target_all = []
        pbar = tqdm(self.loader[split], desc=f'Eval {split.capitalize()}', total=len(self.loader[split]),
                    **pbar_setting)
        #print(f'#IN#{pbar}')
        for data in pbar:
            #print('#IN#',data)
            #print('#IN#for')
            data: Batch = data.to(self.config.device)

            mask, targets = nan2zero_get_mask(data, split, self.config)
            
            if mask is None:
                return stat
            node_norm = torch.ones_like(targets,
                                        device=self.config.device) if self.config.model.model_level == 'node' else None
            #print('#IN#qqq')
            data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                                 self.model.training,
                                                                                 self.config, model=self.model)
            #print('#IN#www')
            
            edge_weight=None
            mask_start_epoch=-1
            shift_str=self.config.dataset.shift_type
            domain=self.config.dataset.domain
            dataset_name=self.config.dataset.dataset_name
            #if self.config.dataset.dataset_name=='GOODWebKB':
            #    mask_start_epoch=50
            if self.config.train.epoch>mask_start_epoch:
                if self.config.use_inv_edge_mask and self.config.ood.ood_alg=='CIA':
                    if self.config.ood.extra_param[1]==1: # CIA-LRA
                        node_features=self.edge_mask_GNN.get_embed(x=data.x, edge_index=data.edge_index, edge_weight=None)
                        if self.config.model.model_name in ['CIAGAT', 'GAT']:
                            edge_weight=compute_edge_mask_sigmoid(node_features=node_features, edges=data.edge_index)
                        elif self.config.model.model_name in ['CIAGCN', 'GCN']:
                            
                            if dataset_name=='GOODCBAS':
                                edge_weight=compute_edge_mask_sigmoid(node_features=node_features, edges=data.edge_index)
                            elif dataset_name=='GOODArxiv' and domain=='time' and shift_str=='concept':
                                edge_weight=compute_edge_mask_sigmoid(node_features=node_features, edges=data.edge_index)
                            else:
                                edge_weight=compute_edge_mask_normalization(node_features=node_features, edges=data.edge_index)
                            #edge_weight=compute_edge_mask_sigmoid(node_features=node_features, edges=data.edge_index)
                            
                            

            model_output = self.model(data=data, edge_weight=edge_weight, ood_algorithm=self.ood_algorithm)
            #print('#IN#eee')
            raw_preds = self.ood_algorithm.output_postprocess(model_output)
            if self.config.ood.ood_alg=='sp':
                idx=torch.where(data.env_id>-1)[0]
                raw_preds=raw_preds[idx]
                loss = torch.nn.functional.cross_entropy(raw_preds, targets.long(), reduction='none') * mask
            #print('#IN#rrr')
            # --------------- Loss collection ------------------
            else:
                loss: torch.tensor = self.config.metric.loss_func(raw_preds, targets, reduction='none') * mask
            mask_all.append(mask)
            loss_all.append(loss)
            #print('#IN#ttt')
            # ------------- Score data collection ------------------
            
            if self.config.ood.ood_alg=='sp':
                pred, target = eval_data_preprocess(data.env_id[idx], raw_preds, mask, self.config)
            else:
                pred, target = eval_data_preprocess(data.y, raw_preds, mask, self.config)
            #print('#IN#yyy')
            pred_all.append(pred)
            target_all.append(target)
            #print('#IN#final')
            #print('#IN#outa')
            #print('#IN#outb')

        # ------- Loss calculate -------
        #print(f'#IN# pred {pred}' )
        #print(f'#IN# pred.max(), pred.min() {pred.max()} {pred.min()}' )
        #print(f'#IN# target {target}' )
        loss_all = torch.cat(loss_all)
        #print('#IN#out1')
        mask_all = torch.cat(mask_all)
        #print('#IN#out2')
        stat['loss'] = loss_all.sum() / mask_all.sum()
        #print('#IN#out3')
        # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
        stat['score'] = eval_score(pred_all, target_all, self.config)

        print(f'#IN#\n{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f}\n'
              f'{split.capitalize()} Loss: {stat["loss"]:.4f}')
        #print('#IN#aaa')
        self.model.train()
        #print('#IN#bbb')
        return {'score': stat['score'], 'loss': stat['loss']}

    def load_task(self):
        r"""
        Launch a training or a test.
        """
        if self.task == 'train':
            self.train()

        elif self.task == 'test':

            # config model
            print('#D#Config model and output the best checkpoint info...')
            test_score, test_loss = self.config_model('test')

    def config_model(self, mode: str, load_param=False):
        r"""
        A model configuration utility. Responsible for transiting model from CPU -> GPU and loading checkpoints.
        Args:
            mode (str): 'train' or 'test'.
            load_param: When True, loading test checkpoint will load parameters to the GNN model.

        Returns:
            Test score and loss if mode=='test'.
        """
        self.model.to(self.config.device)
        self.model.train()

        # load checkpoint
        if mode == 'train' and self.config.train.tr_ctn:
            ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'last.ckpt'))
            self.model.load_state_dict(ckpt['state_dict'])
            best_ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'best.ckpt'))
            self.config.metric.best_stat['score'] = best_ckpt['val_score']
            self.config.metric.best_stat['loss'] = best_ckpt['val_loss']
            self.config.train.ctn_epoch = ckpt['epoch'] + 1
            print(f'#IN#Continue training from Epoch {ckpt["epoch"]}...')

        if mode == 'test':
            try:
                ckpt = torch.load(self.config.test_ckpt, map_location=self.config.device)
            except FileNotFoundError:
                print(f'#E#Checkpoint not found at {os.path.abspath(self.config.test_ckpt)}')
                exit(1)
            if os.path.exists(self.config.id_test_ckpt):
                id_ckpt = torch.load(self.config.id_test_ckpt, map_location=self.config.device)
                # model.load_state_dict(id_ckpt['state_dict'])
                print(f'#IN#Loading best In-Domain Checkpoint {id_ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {id_ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {id_ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {id_ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {id_ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {id_ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {id_ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {id_ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {id_ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {id_ckpt["val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {id_ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {id_ckpt["test_loss"].item():.4f}\n')
                print(f'#IN#Loading best Out-of-Domain Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(f'#IN#ChartInfo {id_ckpt["id_test_score"]:.4f} {id_ckpt["test_score"]:.4f} '
                      f'{ckpt["id_test_score"]:.4f} {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')

            else:
                print(f'#IN#No In-Domain checkpoint.')
                # model.load_state_dict(ckpt['state_dict'])
                print(f'#IN#Loading best Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(
                    f'#IN#ChartInfo {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')
            if load_param:
                if self.config.ood.ood_alg != 'EERM':
                    self.model.load_state_dict(ckpt['state_dict'])
                else:
                    self.model.gnn.load_state_dict(ckpt['state_dict'])
            return ckpt["test_score"], ckpt["test_loss"]

    def save_epoch(self, epoch: int, train_stat: dir, id_val_stat: dir, id_test_stat: dir, val_stat: dir,
                   test_stat: dir, config: Union[CommonArgs, Munch]):
        r"""
        Training util for checkpoint saving.

        Args:
            epoch (int): epoch number
            train_stat (dir): train statistics
            id_val_stat (dir): in-domain validation statistics
            id_test_stat (dir): in-domain test statistics
            val_stat (dir): ood validation statistics
            test_stat (dir): ood test statistics
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.ckpt_dir`, :obj:`config.dataset`, :obj:`config.train`, :obj:`config.model`, :obj:`config.metric`, :obj:`config.log_path`, :obj:`config.ood`)

        Returns:
            None

        """
        state_dict = self.model.state_dict() if config.ood.ood_alg != 'EERM' else self.model.gnn.state_dict()
        ckpt = {
            'state_dict': state_dict,
            'train_score': train_stat['score'],
            'train_loss': train_stat['loss'],
            'id_val_score': id_val_stat['score'],
            'id_val_loss': id_val_stat['loss'],
            'id_test_score': id_test_stat['score'],
            'id_test_loss': id_test_stat['loss'],
            'val_score': val_stat['score'],
            'val_loss': val_stat['loss'],
            'test_score': test_stat['score'],
            'test_loss': test_stat['loss'],
            'time': datetime.datetime.now().strftime('%b%d %Hh %M:%S'),
            'model': {
                'model name': f'{config.model.model_name} {config.model.model_level} layers',
                'dim_hidden': config.model.dim_hidden,
                'dim_ffn': config.model.dim_ffn,
                'global pooling': config.model.global_pool
            },
            'dataset': config.dataset.dataset_name,
            'train': {
                'weight_decay': config.train.weight_decay,
                'learning_rate': config.train.lr,
                'mile stone': config.train.mile_stones,
                'shift_type': config.dataset.shift_type,
                'Batch size': f'{config.train.train_bs}, {config.train.val_bs}, {config.train.test_bs}'
            },
            'OOD': {
                'OOD alg': config.ood.ood_alg,
                'OOD param': config.ood.ood_param,
                'number of environments': config.dataset.num_envs
            },
            'log file': config.log_path,
            'epoch': epoch,
            'max epoch': config.train.max_epoch
        }
        if not (config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat[
            'score'] < config.metric.lower_better *
                config.metric.best_stat['score']
                or (id_val_stat.get('score') and (
                        config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
                    'score'] < config.metric.lower_better * config.metric.id_best_stat['score']))
                or epoch % config.train.save_gap == 0):
            return

        

        if not os.path.exists(config.ckpt_dir):
            os.makedirs(config.ckpt_dir)
            print(f'#W#Directory does not exists. Have built it automatically.\n'
                  f'{os.path.abspath(config.ckpt_dir)}')
        saved_file = os.path.join(config.ckpt_dir, f'{epoch}.ckpt')
        torch.save(ckpt, saved_file)
        shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'last.ckpt'))

        # --- In-Domain checkpoint ---
        if id_val_stat.get('score') and (
                config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
            'score'] < config.metric.lower_better * config.metric.id_best_stat['score']):
            config.metric.id_best_stat['score'] = id_val_stat['score']
            config.metric.id_best_stat['loss'] = id_val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'id_best.ckpt'))
            print('#IM#Saved a new best In-Domain checkpoint.')

        # --- Out-Of-Domain checkpoint ---
        # if id_val_stat.get('score'):
        #     if not (config.metric.lower_better * id_val_stat['score'] < config.metric.lower_better * val_stat['score']):
        #         return
        if config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat[
            'score'] < config.metric.lower_better * \
                config.metric.best_stat['score']:
            config.metric.best_stat['score'] = val_stat['score']
            config.metric.best_stat['loss'] = val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'best.ckpt'))
            print('#IM#Saved a new best checkpoint.')
        if config.clean_save:
            os.unlink(saved_file)
            
            
def compute_edge_mask_sigmoid(node_features, edges):
    node_i = node_features[edges[0]]  
    node_j = node_features[edges[1]] 
    dot_products = (node_i * node_j).sum(dim=1) 
    edge_mask = F.sigmoid(dot_products)
    return edge_mask

def compute_edge_mask_normalization(node_features, edges):
    node_i = node_features[edges[0]]  
    node_j = node_features[edges[1]] 
    dot_products = (node_i * node_j).sum(dim=1) 
    edge_mask = (dot_products-dot_products.min())/(dot_products.max()-dot_products.min()) # most values are very small
    return edge_mask