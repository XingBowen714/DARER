from transformers import BertConfig, BertModel, RobertaModel, XLNetModel, AlbertModel, ElectraModel
from torch_geometric.nn import RGCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.decode import ScaledDotProductAttention

class BiGraphEncoder(nn.Module):
    def __init__(self,
                 word_embedding: nn.Embedding,
                 hidden_dim: int,
                 dropout_rate: float,
                 pretrained_model: str,
                 rgcn_num_bases: int):
        """
        Use BiLSTM + rgcn to Encode
        """

        super(BiGraphEncoder, self).__init__()

        # self._remove_user = remove_user
        if pretrained_model != "none":
            self._utt_encoder = UtterancePretrainedModel(hidden_dim, pretrained_model)
        else:
            self._utt_encoder = BiRNNEncoder(word_embedding, hidden_dim, dropout_rate)
        
        self._pretrained_model = pretrained_model
         
        self._dialog_layer_user = RGCN(hidden_dim, dropout_rate, rgcn_num_bases = rgcn_num_bases)

    # Add for loading best model
    def add_missing_arg(self, pretrained_model):
        self._pretrained_model = pretrained_model

    def forward(self, input_w, adj, pad_adj_full_list, mask=None):
        
        if self._pretrained_model != "none":
            hidden_w = self._utt_encoder(input_w, mask)
        else:
            hidden_w = self._utt_encoder(input_w)
        bi_ret = hidden_w

        ret = self._dialog_layer_user(bi_ret, pad_adj_full_list)
        return ret


class BiRNNEncoder(nn.Module):

    def __init__(self,
                 word_embedding: nn.Embedding,
                 hidden_dim: int,
                 dropout_rate: float):

        super(BiRNNEncoder, self).__init__()

        _, embedding_dim = word_embedding.weight.size()
        self._word_embedding = word_embedding

        self._rnn_cell = nn.LSTM(embedding_dim, hidden_dim // 2,
                                 batch_first=True, bidirectional=True)
        self._drop_layer = nn.Dropout(dropout_rate)

    def forward(self, input_w):
        embed_w = self._word_embedding(input_w)
        dropout_w = self._drop_layer(embed_w)

        hidden_list, batch_size = [], input_w.size(0)
        for index in range(0, batch_size):
            batch_w = dropout_w[index]
            encode_h, _ = self._rnn_cell(batch_w)

            pooling_h, _ = torch.max(encode_h, dim=-2)
            hidden_list.append(pooling_h.unsqueeze(0))

        # Concatenate the representations of each sentence in the batch.
        return torch.cat(hidden_list, dim=0)



class RGCN(nn.Module):
    """
    rgcn to model intra- and inter-speaker dependencies
    """

    def __init__(self, hidden_dim, dropout_rate, rgcn_num_bases):
        super(RGCN, self).__init__()

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.relation_type_to_idx = {}

        speaker_from = ['0','1']
        speaker_to = ['0', '1']
        positions = ['-1', '1']
        
        # respectively denote 'past ones','previous one', 'current', 'next one', 'future ones'
        for j in speaker_from:
            for k in speaker_to:
                for m in positions:
                    self.relation_type_to_idx[str(j) + str(k) + str(m)] = len(self.relation_type_to_idx)
        
        self.relation_type_to_idx['pad'] = len(self.relation_type_to_idx)   
        self.num_relations = len(self.relation_type_to_idx)

        self.edgeatt = ScaledDotProductAttention(hidden_dim)

        if rgcn_num_bases > 0:
            self.RGCN = RGCNConv(hidden_dim, hidden_dim, self.num_relations, num_bases = rgcn_num_bases)
        else:
            self.RGCN = RGCNConv(hidden_dim, hidden_dim, self.num_relations)
   
    # Add for loading best model
    def add_missing_arg(self, layer=2):
        self._dialog_layer.add_missing_arg(layer)

    def batch_graphify(self,features, pad_adj_full_list):
        
        node_features, edge_index, edge_norm, edge_type = [], [], [], []
        batch_size = features.size(0)
        length_sum = 0
        edge_ind = []
        edge_index_lengths = []
        edge_norm = []

        #mask = torch.LongTensor(pad_adj_full_list).cuda()
        #atts = self.edgeatt(features, features, mask)

        for j in range(batch_size):
            
            cur_len = features.size(1)
            assert len(pad_adj_full_list[j]) == cur_len
            
            node_features.append(features[j])
            perms = self.edge_perms(pad_adj_full_list[j])
            perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
            length_sum += cur_len
            edge_index_lengths.append(len(perms))
            
            
            for item, item_rec in zip(perms, perms_rec):
               
                edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
                #edge_norm.append(edge_weights[j][item[0], item[1]])
                #edge_norm.append(atts[j][item[0], item[1]])
                
                if not pad_adj_full_list[j][item[0]][item[1]]:
                    edge_type.append(self.relation_type_to_idx['pad'])
                    continue
                
                if item[0]%2 == 0:
                    task1 = '0'
                else:
                    task1 = '1'
                
                if item[1]%2 == 0:
                    task2 = '0'
                else:
                    task2 = '1'
                
                if item[0] >= item[1]:
                    position = '-1'
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

    def edge_perms(self,pad_adj_full_list_item, window_past=-1, window_future=-1):
        """
        Method to construct the edges of a graph (a utterance) considering the past and future window.
        return: list of tuples. tuple -> (vertice(int), neighbor(int))
        """
        
        all_perms = set()
        
        for i in range(len(pad_adj_full_list_item)):
            
            perms = set()
            
            for j in range(len(pad_adj_full_list_item[i])):
                if pad_adj_full_list_item[i][j]:
                    perms.add((i,j))
                elif i==j:
                    perms.add((i,j))
            all_perms = all_perms.union(perms)
        
        return list(all_perms)
    
    def forward(self, graph_input, pad_adj_full_list):
        
        node_features, edge_index, edge_norm, edge_type, edge_index_lengths = self.batch_graphify(
            graph_input, pad_adj_full_list)
        
        drp = self.RGCN(node_features, edge_index, edge_type) #drp=dialog_representation
        drp = drp.reshape(graph_input.size(0),-1, graph_input.size(2)) 
       
        # chunk into sent and act representation
        drp = self.dropout_layer(drp)
        return drp

class UtterancePretrainedModel(nn.Module):
    HIDDEN_DIM = 768

    def __init__(self, hidden_dim, pretrained_model):
        super(UtterancePretrainedModel, self).__init__()
        self._pretrained_model = pretrained_model

        if pretrained_model == "bert":
            self._encoder = BertModel.from_pretrained("bert-base-uncased")
        elif pretrained_model == "roberta":
            self._encoder = RobertaModel.from_pretrained("roberta-base")
        elif pretrained_model == "xlnet":
            self._encoder = XLNetModel.from_pretrained("xlnet-base-cased")
        elif pretrained_model == "albert":
            self._encoder = AlbertModel.from_pretrained("albert-base-v2")
        elif pretrained_model == "electra":
            self._encoder = ElectraModel.from_pretrained("google/electra-base-discriminator")
        else:
            assert False, "Something wrong with the parameter --pretrained_model"

        self._linear = nn.Linear(UtterancePretrainedModel.HIDDEN_DIM, hidden_dim)

    def forward(self, input_p, mask):
        cls_list = []

        for idx in range(0, input_p.size(0)):
            if self._pretrained_model == "electra":
                cls_tensor = self._encoder(input_p[idx], attention_mask=mask[idx])[0]
            else:
                cls_tensor, op2 = self._encoder(input_p[idx], attention_mask=mask[idx])
        #    print(cls_tensor, op2)
            cls_tensor = cls_tensor[:, 0, :]
            linear_out = self._linear(cls_tensor.unsqueeze(0))
            cls_list.append(linear_out)
        return torch.cat(cls_list, dim=0)
