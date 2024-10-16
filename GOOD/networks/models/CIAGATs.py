import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import GATConv

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier

import torch_geometric

# ... 其他导入 ...

@register.model_register
class CIAGAT(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = GATFeatExtractor(config)
        self.classifier = Classifier(config)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        rep = self.feat_encoder(*args, **kwargs)
        #no_center_rep = self.feat_encoder_no_center(*args, **kwargs)
        pred = self.classifier(rep)
        return pred, rep
    
    def get_embed(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        embed=self.feat_encoder(x, edge_index, None, edge_weight=edge_weight)
        return embed

class GATFeatExtractor(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GATFeatExtractor, self).__init__(config)
        self.encoder = GATEncoder(config)

    def forward(self, *args, **kwargs):
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        out_readout = self.encoder(x, edge_index, edge_weight, batch)
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

    def forward(self, x, edge_index, edge_weight, batch):
        if self.config.dataset.dataset_name!='GOODCora':
            post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_weight))))#
            for i, (conv, batch_norm, relu, dropout) in enumerate(
                    zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
                post_conv = batch_norm(conv(post_conv, edge_index, edge_weight))
                if i < len(self.convs) - 1:
                    post_conv = relu(post_conv)
                post_conv = dropout(post_conv)

            out_readout = self.readout(post_conv, batch)
        else:
            x = self.conv1(x, edge_index, edge_weight)
            for conv in self.convs:
                x = conv(x, edge_index, edge_weight)
            out_readout = self.readout(x, batch)
        return out_readout

# ... 其他代码 ...
