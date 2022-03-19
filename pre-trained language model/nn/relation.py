import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv



class DTRR(nn.Module):
    """
    Dual-task Reasoning Relational Graph Networks
    """

    def __init__(self, hidden_dim, dropout_rate, rgcn_num_bases, num_relations):
        super(DTRR, self).__init__()

        self._sent_linear = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )
        self._act_linear = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim

        if rgcn_num_bases > 0:
            self.RGCN = RGCNConv(hidden_dim, hidden_dim, self.num_relations, num_bases = rgcn_num_bases)
        else:
            self.RGCN = RGCNConv(hidden_dim, hidden_dim, self.num_relations)
    # Add for loading best model
    def add_missing_arg(self, layer=2):
        self._dialog_layer.add_missing_arg(layer)

    def forward(self, input_s, input_a, edge_index, edge_type):
        graph_input = torch.cat([input_s, input_a], dim=1)
        node_features = torch.reshape(graph_input, [-1, self.hidden_dim])
        dtrp = self.RGCN(node_features, edge_index, edge_type) #dtrp=dual_task_representation
        dtrp = dtrp.reshape(input_s.size(0),-1, input_s.size(2) ) 
        # chunk into sent and act representation
        dtrp = self.dropout_layer(dtrp)
        sent, act = torch.chunk(dtrp, 2, dim=1)
        return sent, act

