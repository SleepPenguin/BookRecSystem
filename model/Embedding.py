# 对各个分立特征tensor进行拼接处理，序列特征tensor取全序列对应平均值
import sys
import torch
import torch.nn as nn
from utils.features import SparseFeature, SequenceFeature

sys.path.append('..')


class InputMask(nn.Module):
    """Return inputs mask from given features

    Shape:
        - Input: 
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list or SparseFeature or SequenceFeature): Note that the elements in features are either all instances of SparseFeature or all instances of SequenceFeature.
        - Output: 
            - if input Sparse: `(batch_size, num_features)`
            - if input Sequence: `(batch_size, num_features_seq, seq_length)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, features):
        mask = []
        if not isinstance(features, list):
            features = [features]
        for fea in features:
            fea_mask = x[fea.name].long() != -1
            mask.append(fea_mask.unsqueeze(1).float())
        return torch.cat(mask, dim=1)



class AveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
        non_padding_length = mask.sum(dim=-1)
        return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


class EmbeddingLayer(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0
        for fea in features:
            self.embed_dict[fea.name] = fea.get_embedding_layer()

    def forward(self, x, features):
        sparse_emb = []
        for fea in features:
            if isinstance(fea, SparseFeature):
                sparse_emb.append(self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1))
            elif isinstance(fea, SequenceFeature):
                pooling_layer = AveragePooling()
                fea_mask = InputMask()(x, fea)
                # shared specific sparse feature embedding
                sparse_emb.append(pooling_layer(self.embed_dict[fea.shared_with](x[fea.name].long()), fea_mask).unsqueeze(1))
        # [batch_size, num_features, embed_dim]
        sparse_emb = torch.cat(sparse_emb, dim=2)

        return sparse_emb.flatten(start_dim=1)
