import torch
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
import os
import torch_geometric
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time
from collections import deque
import numpy  as np
from scipy import stats
@register.ood_alg_register
class sp(BaseOODAlg):
    r"""
    Implementation of the baseline ERM

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(sp, self).__init__(config)


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

    def vis_sp(self, A, Y, rep, env_id=None):
        '''
        compute CIA loss with reweighting
        if env_id!=None, then use ground truth env label, and only condsider pairs 
        with different env labels'''
        if Y.dim()>1:
            Y.squeeze_(1)
        Y=Y.long()

        num_classes=int(torch.max(Y))+1
        labels_one_hot = torch.nn.functional.one_hot(Y, num_classes)

        neighbor_label_counts = torch.matmul(A, labels_one_hot.float()) # dim=[N,C], for computing num_diff_diff

        Ak=self.cal_Ak(A,k=4)

        del labels_one_hot # save memory
        for c in range(num_classes):

            y_id=torch.where(Y==c)[0]
            #print(f'#in#{y_id}')
            #print(f'#in#rep {rep}')
            rep_c=rep[y_id]
            num_cur_c=y_id.shape[0]
            print(f'#in# num_cur_c {num_cur_c}')
            
            neighbor_label_counts_c=neighbor_label_counts[y_id]
            hetero_num_c=torch.cat((neighbor_label_counts_c[:,:c],neighbor_label_counts_c[:,c+1:]), dim=1) # exclude current class
            
            '''
            hetero_label_dis=[]
            rep_dis=[]'''
            '''for i in range(num_cur_c):
                for j in range(num_cur_c):
                    hetero_label_dis.append(int(torch.sum(torch.abs(hetero_num_c[i]-hetero_num_c[j]), dim=-1).cpu().detach().numpy()))
                    rep_dis.append(float(torch.norm(rep_c[i]-rep_c[j]).cpu().detach().numpy()))
                #save_path=f'codes/GOOD/visualization/sp-feature-distance_vs_label-distance/cora/word/cov/by_node/class-{c}/node-{i}/all_homo_.jpg'
            save_path=f'codes/GOOD/visualization/sp-feature-distance_vs_label-distance/arxiv/degree/cov/by_class/1l-class-{c}.jpg'    '''


            
            # sp, by homo
            '''shift_str=self.config.dataset.shift_type
            domain=self.config.dataset.domain
            dataset_name=self.config.dataset.dataset_name
            for num_homo in range(8):
                hetero_label_dis=[]
                rep_dis=[]
                for i in range(num_cur_c):
                    for j in range(i+1,num_cur_c):
                        if neighbor_label_counts_c[j][c]==num_homo and neighbor_label_counts_c[i][c]==num_homo and Ak[y_id[i]][y_id[j]]<torch.inf:
                            hetero_label_dis.append(int(torch.sum(torch.abs(hetero_num_c[i]-hetero_num_c[j]), dim=-1).cpu().detach().numpy()))
                            rep_dis.append(torch.norm(rep_c[i]-rep_c[j]).cpu().detach().numpy())

                save_path=os.path.join('codes/GOOD/visualization/sp-feature-distance_vs_label-distance',dataset_name,domain,shift_str,'pred-env-label_local_by_homo',f'class-{c}/all_homo_{num_homo}.jpg')
                
                #save_path=f'codes/GOOD/visualization/sp-feature-distance_vs_label-distance/arxiv/degree/cov/by_class/1l-class-{c}.jpg'   

                # 将 x 和 y 值配对，并按 x 值分类
                xy_pairs = defaultdict(list)
                for x, y in zip(hetero_label_dis, rep_dis):
                    xy_pairs[x].append(y)

                # 准备散点图的 x 值和 y 值
                scatter_x = []
                scatter_y = []

                # 对于每个独特的 x 值，最多选取 50 个 y 值
                for x, y_list in xy_pairs.items():
                    if len(y_list) > 50:
                        y_list = random.sample(y_list, 50)
                    scatter_x.extend([x] * len(y_list))
                    scatter_y.extend(y_list)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                if not scatter_x or len(np.unique(scatter_x))==1:
                    continue
                a, b, r_value, p_value, std_err = stats.linregress(scatter_x, scatter_y)
                # 生成拟合线的 x 和 y 值
                fit_x = np.arange(-1,np.max(scatter_x)+1)
                fit_y = a * fit_x + b

                plt.scatter(scatter_x, scatter_y)
                plt.plot(fit_x, fit_y, color='red', label='Linear fit')  # 添加拟合线
                plt.title(f'Class {c}')
                plt.xlabel('Heterophilous Label distance')
                plt.ylabel('Spurious Feature Distance')
                #plt.ylim(0, 20)
                plt.savefig(save_path)
                plt.clf()'''
            
            shift_str=self.config.dataset.shift_type
            domain=self.config.dataset.domain
            dataset_name=self.config.dataset.dataset_name
            hetero_label_dis=[]
            rep_dis=[]
            for i in range(num_cur_c):
                for j in range(i+1,num_cur_c):
                    if  Ak[y_id[i]][y_id[j]]<torch.inf:
                        hetero_label_dis.append(int(torch.sum(torch.abs(hetero_num_c[i]-hetero_num_c[j]), dim=-1).cpu().detach().numpy()))
                        rep_dis.append(torch.norm(rep_c[i]-rep_c[j]).cpu().detach().numpy())

            save_path=os.path.join('codes/GOOD/visualization/sp-feature-distance_vs_label-distance',dataset_name,domain,shift_str,'pred-env-label_local',f'class-{c}.jpg')
            
            #save_path=f'codes/GOOD/visualization/sp-feature-distance_vs_label-distance/arxiv/degree/cov/by_class/1l-class-{c}.jpg'   

            # 将 x 和 y 值配对，并按 x 值分类
            xy_pairs = defaultdict(list)
            for x, y in zip(hetero_label_dis, rep_dis):
                xy_pairs[x].append(y)

            # 准备散点图的 x 值和 y 值
            scatter_x = []
            scatter_y = []

            # 对于每个独特的 x 值，最多选取 50 个 y 值
            for x, y_list in xy_pairs.items():
                if len(y_list) > 50:
                    y_list = random.sample(y_list, 50)
                scatter_x.extend([x] * len(y_list))
                scatter_y.extend(y_list)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            if not scatter_x or len(np.unique(scatter_x))==1:
                continue
            a, b, r_value, p_value, std_err = stats.linregress(scatter_x, scatter_y)
            # 生成拟合线的 x 和 y 值
            fit_x = np.arange(-1,np.max(scatter_x)+1)
            fit_y = a * fit_x + b

            plt.scatter(scatter_x, scatter_y)
            plt.plot(fit_x, fit_y, color='red', label='Linear fit')  # 添加拟合线
            plt.title(f'Class {c}')
            plt.xlabel('Heterophilous Label distance')
            plt.ylabel('Spurious Feature Distance')
            #plt.ylim(0, 20)
            plt.savefig(save_path)
            plt.clf()
                
            '''for i in range(num_cur_c):
                key = str(sp_key[i].tolist())
                if neighbor_label_counts_c[i] in d: 
                    
                    d[key].append(rep_c[i])
                else:
                    d[key]=[]
                    d[key].append(rep_c[i])
            for k,v in d.items():
                print(f'#in# k={k}, v={len(v)}')'''
            
            '''# inv
            shift_str=self.config.dataset.shift_type
            domain=self.config.dataset.domain
            dataset_name=self.config.dataset.dataset_name
            homo_label_dis=[]
            rep_dis=[]
            for i in range(num_cur_c):
                for j in range(i+1,num_cur_c):
                    if int(torch.sum(torch.abs(hetero_num_c[i]-hetero_num_c[j]), dim=-1).cpu().detach().numpy())<1 and Ak[y_id[i]][y_id[j]]<torch.inf:
                        homo_label_dis.append(int(torch.abs(neighbor_label_counts_c[i][c]-neighbor_label_counts_c[j][c]).cpu().detach().numpy()))
                        rep_dis.append(torch.norm(rep_c[i]-rep_c[j]).cpu().detach().numpy())

            save_path=os.path.join('codes/GOOD/visualization/inv-feature-distance_vs_label-distance',dataset_name,domain,shift_str,'local_hetero_similar',f'class-{c}.jpg')
                
                #save_path=f'codes/GOOD/visualization/sp-feature-distance_vs_label-distance/arxiv/degree/cov/by_class/1l-class-{c}.jpg'   

            # 将 x 和 y 值配对，并按 x 值分类
            xy_pairs = defaultdict(list)
            for x, y in zip(homo_label_dis, rep_dis):
                xy_pairs[x].append(y)

            # 准备散点图的 x 值和 y 值
            scatter_x = []
            scatter_y = []

            # 对于每个独特的 x 值，最多选取 50 个 y 值
            for x, y_list in xy_pairs.items():
                if len(y_list) > 50:
                    y_list = random.sample(y_list, 50)
                scatter_x.extend([x] * len(y_list))
                scatter_y.extend(y_list)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            if not scatter_x or len(np.unique(scatter_x))==1:
                continue
            a, b, r_value, p_value, std_err = stats.linregress(scatter_x, scatter_y)
            # 生成拟合线的 x 和 y 值
            fit_x = np.arange(-1,np.max(scatter_x)+1)
            fit_y = a * fit_x + b

            plt.scatter(scatter_x, scatter_y)
            plt.plot(fit_x, fit_y, color='red', label='Linear fit')  # 添加拟合线
            plt.title(f'Class {c}')
            plt.xlabel('Homophilous Label distance')
            plt.ylabel('Spurious Feature Distance')
            #plt.ylim(0, 20)
            plt.savefig(save_path)
            plt.clf()'''
        #print(f"#IN#t_end-t0:{t_end-t0}")
        return
    
    def generate_random_paths(self, edge_index, num_paths=10, path_length=5):
        paths = []
        for _ in range(num_paths):
            start_node = random.randint(0, edge_index.max())
            path = [start_node]
            for _ in range(path_length - 1):
                neighbors = edge_index[1][edge_index[0] == path[-1]]
                if len(neighbors) == 0:
                    break
                next_node = neighbors[random.randint(0, len(neighbors) - 1)]
                path.append(next_node.item())
            paths.append(path)
        return paths



    '''def generate_class_constrained_paths(self, edge_index, labels, class_label, num_paths=10, path_length=5):
        paths = []
        nodes_of_class = (labels == class_label).nonzero(as_tuple=True)[0]

        for _ in range(num_paths):
            start_node = nodes_of_class[random.randint(0, len(nodes_of_class) - 1)].item()
            visited = set([start_node])
            queue = deque([(start_node, [start_node])])

            while queue:
                current_node, path = queue.popleft()
                if len(path) == path_length:
                    paths.append(path)
                    break

                neighbors = edge_index[1][edge_index[0] == current_node]
                for next_node in neighbors:
                    if next_node.item() in visited or labels[next_node] != class_label:
                        continue
                    visited.add(next_node.item())
                    queue.append((next_node.item(), path + [next_node.item()]))
            
            if len(paths) == num_paths:
                break

        return paths'''



    def generate_class_constrained_paths(self, edge_index, labels, class_label, num_paths=10, num_path_nodes=20):
        paths = []
        nodes_of_class = (labels == class_label).nonzero(as_tuple=True)[0].tolist()

        # 执行广度优先搜索并记录每个节点的BFS距离
        def bfs(start_node):
            bfs_distances = {start_node: 0}
            #path=[start_node]
            dis_path=defaultdict(list)
            queue = deque([start_node])

            print(f'#in#class{class_label}')
            while queue:
                if len(dis_path)>num_path_nodes:
                    break
                current_node = queue.popleft()
                current_distance = bfs_distances[current_node]
                neighbors = edge_index[1][edge_index[0] == current_node]
                
                for next_node in neighbors:
                    if next_node.item() not in bfs_distances:
                        bfs_distances[next_node.item()] = current_distance + 1
                        queue.append(next_node.item())
                
                    if next_node.item() in nodes_of_class: # node from cur class
                        dis_path[current_distance + 1].append(next_node.item())
                        
                    
            return dis_path

        for _ in range(num_paths):
            start_node = random.choice(nodes_of_class)
            dis_path = bfs(start_node)

            # 生成路径
            path_nodes = [start_node]
            path_dis=[0]
            for dis, nodes in dis_path.items():
                if len(nodes)>0:
                    path_nodes.append(random.choice(nodes))
                    path_dis.append(dis)
            paths.append((path_nodes, path_dis))
        return paths





    def compute_path_representations(self, rep, paths):
        path_reps = []
        for (path_nodes, path_dis) in paths:
            print(f'#in#path_nodes={path_nodes}')
            reps = rep[path_nodes].detach().cpu()
            distances = torch.norm(reps - reps[0], dim=1)
            path_reps.append(distances)
        return path_reps



    def plot_representation_changes(self, path_reps_inv, path_reps_sp, c, paths):
        plt.figure(figsize=(12, 8))
        t=str(time.time())
        #save_path=f'codes/GOOD/visualization/feature-changing-rate/cora/word/cov/rate-'+t+'.jpg'
        save_path=f'codes/GOOD/visualization/feature-changing-rate/cora/word/cov/rate-'+'class-'+str(c)+'-'+t+'.jpg'
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        for i, (rep_inv, rep_sp, (path_nodes, path_dis)) in enumerate(zip(path_reps_inv, path_reps_sp, paths)):
            plt.subplot(2, 5, i + 1)
            plt.plot(path_dis, rep_inv, label='Invariant Feature')
            plt.plot(path_dis, rep_sp, label='Spurious Feature')
            plt.title(f'Path {i+1}')
            plt.xlabel('Path Length')
            plt.ylabel('L2 Distance from Start')
            plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)



    def vis_rate(self, rep_sp, rep_inv, edge_index, Y, c):
        #paths=self.generate_random_paths(edge_index, num_paths=10, path_length=20)
        paths = self.generate_class_constrained_paths(edge_index, Y, c, num_paths=10)
        print(f'#in#paths: {paths}')
        path_reps_inv = self.compute_path_representations(rep_inv, paths)
        path_reps_sp = self.compute_path_representations(rep_sp, paths)
        self.plot_representation_changes(path_reps_inv, path_reps_sp, c, paths)
        return
    
    

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:

        # Add this at the end of the function before returning the loss
        '''if config.train.epoch<40 and config.train.batch_id>config.train.num_batches-2 and config.train.epoch%3==0:
            save_path = '/data1/qxwang/codes/GOOD/visualization/ERM_t-SNE'
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f'ERM_{config.dataset.dataset_name}_tsne_epoch_{config.train.epoch}.png')
            self.visualize_tsne(self.rep, data.y, data.env_id, file_path, drop_classes=True, num_classes_to_visualize=8)'''
        #print(f'#in# kwargs {kwargs}')
        self.mean_loss = loss.sum() / mask.sum()
       
        if config.ood.extra_param[0]==1: # sp feature distance vs hetero label distance;
            pretrained_sp_model=kwargs['pretrained_model']
            pred, rep=pretrained_sp_model(data=data, edge_weight=kwargs['edge_weight'], ood_algorithm=kwargs['ood_algorithm'])
            A=torch_geometric.utils.to_dense_adj(data.edge_index).squeeze(0)
            self.vis_sp(A, data.y,rep)
        
        elif config.ood.extra_param[0]==2: # change rate of sp and inv features
            pretrained_sp_model=kwargs['pretrained_model']
            pretrained_inv_model=kwargs['pretrained_inv_model']
            pred, rep_sp=pretrained_sp_model(data=data, edge_weight=kwargs['edge_weight'], ood_algorithm=kwargs['ood_algorithm'])
            pred, rep_inv=pretrained_inv_model(data=data, edge_weight=kwargs['edge_weight'], ood_algorithm=kwargs['ood_algorithm'])
            for c in range(self.config.dataset.num_classes):
                self.vis_rate(rep_sp, rep_inv, data.edge_index, data.y, c)
            
        return self.mean_loss
    
'''class TensorDict:
    def __init__(self):
        self.dict = {}

    def __setitem__(self, key, value):
        if isinstance(key, torch.Tensor):
            # 将张量转换为字符串作为字典的键
            key = str(key.tolist())
        self.dict[key] = value

    def __getitem__(self, key):
        if isinstance(key, torch.Tensor):
            
        return self.dict[key]
'''
