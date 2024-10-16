import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import GATConv

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier

import torch_geometric
import torch.nn.functional as F
from torch_sparse import coalesce, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
# ... 其他导入 ...

@register.model_register
class GAT(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = GATFeatExtractor(config)
        self.classifier = Classifier(config)
        #self.convs=self.feat_encoder.encoder.convs

    def forward(self, *args, **kwargs) -> torch.Tensor:
        out_readout = self.feat_encoder(*args, **kwargs)
        out = self.classifier(out_readout)
        return out
    
    def get_embed(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        embed=self.feat_encoder(x, edge_index, None, edge_weight=edge_weight)
        return embed

class GATFeatExtractor(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GATFeatExtractor, self).__init__(config)
        self.encoder = GATEncoder(config)

    def forward(self, *args, **kwargs):
        x, edge_index, _, batch = self.arguments_read(*args, **kwargs)
        #print(f'#in# batch={batch}')
        if "edge_weight" in kwargs.keys():
            edge_weight=kwargs["edge_weight"]
        else:
            edge_weight=None
        
        out_readout = self.encoder(x, edge_index, None, edge_weight)
        return out_readout

class GATEncoder(BasicEncoder):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GATEncoder, self).__init__(config)
        num_layers = config.model.model_layer
        self.conv1 = GATConv(config.dataset.dim_node, config.model.dim_hidden, edge_dim=1)
        self.convs = nn.ModuleList(
            [GATConv(config.model.dim_hidden, config.model.dim_hidden, edge_dim=1) for _ in range(num_layers - 1)]
        )
        self.config=config

    def forward(self, x, edge_index, batch, edge_weight=None):
        if self.config.dataset.dataset_name!='GOODCora':
            post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))#
            for i, (conv, batch_norm, relu, dropout) in enumerate(
                    zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
                post_conv = batch_norm(conv(post_conv, edge_index))
                if i < len(self.convs) - 1:
                    post_conv = relu(post_conv)
                post_conv = dropout(post_conv)

            out_readout = self.readout(post_conv, batch)
        else:
            x = self.conv1(x, edge_index)
            for conv in self.convs:
                x = conv(x, edge_index)
            out_readout = self.readout(x, batch)
        return out_readout