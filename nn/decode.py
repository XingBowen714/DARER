import torch.nn as nn
import torch
from nn.relation import DTRR
import torch.nn.functional as F

class RelationDecoder(nn.Module):

    def __init__(self,
                 num_sent: int,
                 num_act: int,
                 hidden_dim: int,
                 num_layer: int,
                 dropout_rate: float,
                 rgcn_num_base: int,
                 stack_num: int
                 ):
        super(RelationDecoder, self).__init__()
        
        self._num_layer = num_layer
   
        self.stack_num = stack_num
        self._sent_layer_dict = nn.ModuleDict()
        self._act_layer_dict = nn.ModuleDict()

        # First with a BiLSTM layer to get the initial representation of SC and DAR
        self._sent_layer_dict.add_module(
                str(0), BiLSTMLayer(hidden_dim, dropout_rate)
            )
        self._act_layer_dict.add_module(
                str(0), BiLSTMLayer(hidden_dim, dropout_rate)
            )
        
        # After each calculation, the specified layer will be passed
        for layer_i in range(1, num_layer):
            
            self._sent_layer_dict.add_module(
                str(layer_i), BiLSTMLayer(hidden_dim, dropout_rate)
            )
            self._act_layer_dict.add_module(
                str(layer_i), BiLSTMLayer(hidden_dim, dropout_rate)
            )
        
        self.hidden_dim = hidden_dim
        self.relation_type_to_idx = {}

        task1_id = [0,1] #0 denotes 'SC', 1 denotes 'AR'
        task2_id = [0,1]

        position_relations = [-1, 0, 1] 
        # respectively denote 'previous', 'current', 'future'
        # respectively denote 'past ones','previous one', 'current', 'next one', 'future ones'
       
        for j in task1_id:
            for k in task2_id:
                for m in position_relations:
                    self.relation_type_to_idx[str(j) + str(k) + str(m)] = len(self.relation_type_to_idx)
        
        self.relation_type_to_idx['pad'] = len(self.relation_type_to_idx)
        self.num_relations = len(self.relation_type_to_idx)

        self.edgeatt = ScaledDotProductAttention(hidden_dim)

        self.senti_labelembedding = nn.Parameter(torch.zeros((num_sent, hidden_dim)).float(), requires_grad=True)
        var = 2. / (self.senti_labelembedding .size(0) + self.senti_labelembedding .size(1))
        self.senti_labelembedding.data.normal_(0, var)

        self.act_labelembedding = nn.Parameter(torch.zeros((num_act, hidden_dim)).float(), requires_grad=True)
        var = 2. / (self.act_labelembedding .size(0) + self.act_labelembedding .size(1))
        self.act_labelembedding.data.normal_(0, var)

        self._relate_layer = DTRR(hidden_dim, dropout_rate, rgcn_num_base, self.num_relations)

        self.dropout_linear = nn.Dropout(dropout_rate)
        self._sent_linear = nn.Linear(hidden_dim, num_sent)
        self._act_linear = nn.Linear(hidden_dim, num_act)

    # Add for loading best model
    def add_missing_arg(self, layer=2):
        self._relate_layer.add_missing_arg(layer)


    def batch_graphify(self,features, pad_adj_R_list):
        
        node_features, edge_index, edge_norm, edge_type = [], [], [], []
        batch_size = features.size(0)
        length_sum = 0
        edge_ind = []
        edge_index_lengths = []
        edge_norm = []

        #mask = torch.LongTensor(pad_adj_R_list).cuda()
        #atts = self.edgeatt(features, features, mask)

        for j in range(batch_size):
            
            cur_len = features.size(1)
            assert len(pad_adj_R_list[j]) == cur_len
            node_features.append(features[j])
            perms = self.edge_perms(pad_adj_R_list[j])
            perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
            length_sum += cur_len
            edge_index_lengths.append(len(perms))
            
            
            for item, item_rec in zip(perms, perms_rec):
                
                edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
                #edge_norm.append(edge_weights[j][item[0], item[1]])
                #edge_norm.append(atts[j][item[0], item[1]])
                
                if not pad_adj_R_list[j][item[0]][item[1]]:
                    edge_type.append(self.relation_type_to_idx['pad'])
                    continue
                
                if item[0] < cur_len / 2:
                    task1 = '0'
                else:
                    task1 = '1'
                
                if item[1] < cur_len / 2:
                    task2 = '0'
                else:
                    task2 = '1'
                
                if item[0]%(cur_len/2) > item[1]%(cur_len/2):
                    position = '-1'
                elif item[0]%(cur_len/2) == item[1]%(cur_len/2):
                    position = '0'
                else:
                    position = '1'
               
                try:
                    edge_type.append(self.relation_type_to_idx[task1+task2+position])
                except Exception as e:
                    print(e)
                    print(item, cur_len)
                    print(self.relation_type_to_idx)
                    assert 9==0
        
        
        node_features = torch.cat(node_features, dim=0).cuda()  # [E, D_g]
        edge_index = torch.stack(edge_index).t().contiguous().cuda()  # [2, E]
        #edge_norm = torch.stack(edge_norm).cuda()  # [E]
        edge_type = torch.tensor(edge_type).long().cuda()  # [E]
        edge_index_lengths = torch.tensor(edge_index_lengths).long().cuda()  # [B]

        return node_features, edge_index, edge_norm, edge_type, edge_index_lengths

    def edge_perms(self,pad_adj_R_list_item, window_past=-1, window_future=-1):
        """
        Method to construct the edges of a graph (a utterance) considering the past and future window.
        return: list of tuples. tuple -> (vertice(int), neighbor(int))
        """
       
        all_perms = set()
       
        for i in range(len(pad_adj_R_list_item)):
            perms = set()
            
            for j in range(len(pad_adj_R_list_item[i])):
                if pad_adj_R_list_item[i][j]:
                    perms.add((i,j))
                elif i==j:
                    perms.add((i,j))
            
            all_perms = all_perms.union(perms)
       
        return list(all_perms)
    

    def forward(self, input_h, len_list, pad_adj_R_list):
        
        sent_h = self._sent_layer_dict["0"](input_h)
        act_h = self._act_layer_dict["0"](input_h)

        graph_input = torch.cat([sent_h, act_h], dim=1)
            
        node_features, edge_index, edge_norm, edge_type, edge_index_lengths = self.batch_graphify(
            graph_input, pad_adj_R_list)
        

        sent_logits, act_logits = [], []
        sent_hiddens, act_hiddens = [], [] 
        
        #residual connection and predict
        linear_s = self._sent_linear(sent_h + input_h)
        linear_a = self._act_linear(act_h + input_h)
        
        p_senti=F.softmax(linear_s, dim = -1)
        p_act=F.softmax(linear_a, dim = -1)

        #record logits history for loss calculation
        sent_logits.append(linear_s)
        act_logits.append(linear_a)
       
        sent_hiddens.append(sent_h)
        act_hiddens.append(act_h)
        
        
        for i in range(self.stack_num):
           
            res_s, res_a = sent_h, act_h
           
            #label embeddings
            senti_lemb = torch.matmul(p_senti, self.senti_labelembedding)
            act_lemb = torch.matmul(p_act, self.act_labelembedding)

            res_s = res_s + senti_lemb + act_lemb
            res_a = res_a + senti_lemb + act_lemb
            
            sent_r, act_r = self._relate_layer(res_s, res_a, edge_index, edge_type)

            # stack num layer CAN NOT be change here.
            # we ONLY stack 1 layer in our experiment.
            # We stack different GAT layer in relation.py
            # you can change gat_layer parameter to control the number of gat layer.
            sent_h = self._sent_layer_dict[str(1)](sent_r)
            act_h = self._act_layer_dict[str(1)](act_r)

            #sent_h, act_h = sent_h + input_h, act_h + input_h
            
            #residual connection and predict
            linear_s = self._sent_linear(sent_h + input_h)
            linear_a = self._act_linear(act_h + input_h)
            
            p_senti = F.softmax(linear_s, dim = -1)
            p_act = F.softmax(linear_a, dim = -1)
            
            #record logits history for loss calculation
            sent_logits.append(linear_s)
            act_logits.append(linear_a)            
            sent_hiddens.append(sent_h)
            act_hiddens.append(act_h)
        
        return sent_logits, act_logits, sent_hiddens, act_hiddens

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, hidden_dim):
        super().__init__()
        self.temperature = hidden_dim**0.5

        #self.dropout = nn.Dropout(attn_dropout)
        self.q_l = nn.Linear(hidden_dim, hidden_dim)
        self.k_l = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, mask=None):

        attn = torch.matmul(self.q_l(q) / self.temperature, self.k_l(k).transpose(1, 2))
        
        #print(attn.size(), mask.size())
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        return attn


class UniLSTMLayer(nn.Module):

    def __init__(self, hidden_dim, dropout_rate):
        super(UniLSTMLayer, self).__init__()

        self._rnn_layer = nn.LSTM(
            hidden_dim, hidden_size=hidden_dim,
            batch_first=True, bidirectional=False
        )
        
        self._drop_layer = nn.Dropout(dropout_rate)

    def forward(self, input_h):
        dropout_h = self._drop_layer(input_h)
        return self._rnn_layer(dropout_h)[0]


class BiLSTMLayer(nn.Module):

    def __init__(self, hidden_dim, dropout_rate):
        super(BiLSTMLayer, self).__init__()

        self._rnn_layer = nn.LSTM(
            hidden_dim, hidden_size=hidden_dim // 2,
            batch_first=True, bidirectional=True
        )
        self._drop_layer = nn.Dropout(dropout_rate)

    def forward(self, input_h):
        dropout_h = self._drop_layer(input_h)
        return self._rnn_layer(dropout_h)[0]


class UniLinearLayer(nn.Module):

    def __init__(self, hidden_dim, dropout_rate):
        super(UniLinearLayer, self).__init__()

        self._linear_layer = nn.Linear(hidden_dim, hidden_dim)
        self._drop_layer = nn.Dropout(dropout_rate)

    def forward(self, input_h):
        dropout_h = self._drop_layer(input_h)
        return self._linear_layer(dropout_h)


class LinearDecoder(nn.Module):
    def __init__(self, num_sent: int, num_act: int, hidden_dim: int):
        super(LinearDecoder, self).__init__()
        self._sent_linear = nn.Linear(hidden_dim, num_sent)
        self._act_linear = nn.Linear(hidden_dim, num_act)

    def forward(self, input_h, len_list, adj_re):
        return self._sent_linear(input_h), self._act_linear(input_h)
