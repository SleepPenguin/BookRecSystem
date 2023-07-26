import torch
import math
import numpy as np

def get_auto_embedding_dim(num_classes):
    # reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    return int(np.floor(6 * math.pow(num_classes, 0.25)))

class RandomNormal(object):
    # 返回使用正态分布初始化的嵌入
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, vocab_size, embed_dim):
        embed = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.normal_(embed.weight, self.mean, self.std)
        return embed 


class SparseFeature(object):
    def __init__(self, name, vocab_size, initializer=RandomNormal(0, 0.0001)):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = get_auto_embedding_dim(vocab_size)
        self.initializer = initializer

    def __repr__(self):
        return f'<SparseFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>'

    def get_embedding_layer(self):
        self.embed = self.initializer(self.vocab_size, self.embed_dim)
        return self.embed


class SequenceFeature(object):
    # Note that if you use this feature, you must padding the feature value before training.
    def __init__(self, name, vocab_size, shared_with=None, initializer=RandomNormal(0, 0.0001)):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = get_auto_embedding_dim(vocab_size)
        self.shared_with = shared_with
        self.initializer = initializer

    def __repr__(self):
        return f'<SequenceFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>'

    def get_embedding_layer(self):
        if not hasattr(self, 'embed'):
            self.embed = self.initializer(self.vocab_size, self.embed_dim)
        return self.embed