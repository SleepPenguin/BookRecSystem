import torch
from .Embedding import EmbeddingLayer
from .MLP import MLP
import torch.nn.functional as F


class DSSM(torch.nn.Module):

    def __init__(self, user_features, item_features, user_params, item_params, temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.item_dims = sum([fea.embed_dim for fea in item_features])
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = MLP(self.user_dims, **user_params, dropout=0.2)
        self.item_mlp = MLP(self.item_dims, **item_params, dropout=0.2)
        self.mode = None

    def user_tower(self, x):
        if self.mode == "item":
            return None
        # [batch_size, num_features*deep_dims]
        input_user = self.embedding(x, self.user_features)
        # [batch_size, user_params["dims"][-1]]
        user_embedding = self.user_mlp(input_user)
        user_embedding = F.normalize(user_embedding, p=2, dim=1)  # L2 normalize
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        # [batch_size, num_features*embed_dim]
        input_item = self.embedding(x, self.item_features)
        # [batch_size, item_params["dims"][-1]]
        item_embedding = self.item_mlp(input_item)
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        return item_embedding

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        # calculate cosine score
        y = torch.mul(user_embedding, item_embedding).sum(dim=1)
        y = y / self.temperature
        return torch.sigmoid(y)

