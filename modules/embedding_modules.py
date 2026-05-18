import numpy as np
import torch
import torch.nn as nn
from .data import TemporalGraphData

class EmbeddingModule(nn.Module):
    def __init__(self,
            node_dim:int=32,
            memory_dim:int=32,
            latent_dim:int=32,
            n_layer:int=1
        ):
        super().__init__()
        self.node_dim=node_dim
        self.memory_dim=memory_dim
        self.latent_dim=latent_dim
        self.n_layer=n_layer

    def compute_embedding(self):
        return NotImplemented

class GraphEmbeddingModule(EmbeddingModule):
    def __init__(self,
            node_dim:int=32,
            memory_dim:int=32,
            latent_dim:int=32,
            n_layer:int=1
        ):
        super(GraphEmbeddingModule,self).__init__(
            node_dim,
            memory_dim,
            latent_dim,
            n_layer
        )

    def compute_embedding(self,
            batch_tar,
            batch_t,
            n_layer
        ):
        """
        Recursive implementation of temporal graph attention layers.
        Input:
            batch_tar: [B,]
            batch_t: [B,]
            n_layer
        Output:
            updated batch_tar_ft: [B,latent_dim]
        """
        # Memory 
