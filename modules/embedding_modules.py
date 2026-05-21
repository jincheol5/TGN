import torch
import torch.nn as nn
from .data import TemporalGraphData,MemoryData
from .time_encoding import TimeEncoder
from .attention_modules import TemporalGraphAttention

class EmbeddingModule(nn.Module):
    def __init__(self,
            node_dim:int=32,
            mem_dim:int=32,
            latent_dim:int=32,
            output_dim:int=32,
            time_dim:int=32,
            graph_data:TemporalGraphData=None,
            memory_data:MemoryData=None,
            time_encoder:TimeEncoder=None,
            n_head:int=1,
            n_layer:int=1,
            is_memory:bool=True
        ):
        super().__init__()
        self.node_dim=node_dim
        self.mem_dim=mem_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim
        self.time_dim=time_dim
        self.graph_data=graph_data
        self.memory_data=memory_data
        self.n_head=n_head
        self.n_layer=n_layer
        self.is_memory=is_memory
        self.time_encoder=time_encoder

    def compute_embedding(self):
        return NotImplemented

class GraphAttentionEmbedding(EmbeddingModule):
    def __init__(self,
            node_dim:int=32,
            mem_dim:int=32,
            latent_dim:int=32,
            output_dim:int=32,
            time_dim:int=32,
            graph_data:TemporalGraphData=None,
            memory_data:MemoryData=None,
            time_encoder:TimeEncoder=None,
            n_head:int=1,
            n_layer:int=1,
            is_memory:bool=True
        ):
        super(GraphAttentionEmbedding,self).__init__(
            node_dim=node_dim,
            mem_dim=mem_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            time_dim=time_dim,
            graph_data=graph_data,
            memory_data=memory_data,
            time_encoder=time_encoder,
            n_head=n_head,
            n_layer=n_layer,
            is_memory=is_memory
        )
        # attn module
        layer_0_input_dim=node_dim+mem_dim if self.is_memory else node_dim
        self.attn_layers=torch.nn.ModuleList([
            TemporalGraphAttention(
                input_dim=layer_0_input_dim if idx==0 else output_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                time_dim=time_dim,
                n_head=n_head
            )
        for idx in range(n_layer)])

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
        if n_layer==0:
            if self.is_memory:
                batch_tar_mem=self.memory_data.get_batch_memory(batch_node=batch_tar) # [B,mem_dim]
                batch_tar_ft=self.graph_data.get_batch_node_feature(batch_node=batch_tar) # [B,node_dim]
                batch_tar_ft=torch.concat(
                    [batch_tar_mem,batch_tar_ft],
                    dim=-1
                ) # [B,mem_dim+node_dim]
            else:
                batch_tar_ft=self.graph_data.get_batch_node_feature(batch_node=batch_tar) # [B,node_dim]
            return batch_tar_ft
        else:
            batch_tar_ft=self.compute_embedding(
                batch_tar=batch_tar,
                batch_t=batch_t,
                n_layer=n_layer-1
            )
            embed_data=self.graph_data.get_data_for_embedding(
                batch_tar=batch_tar,
                batch_t=batch_t
            )
            batch_tar_ts=embed_data["batch_tar_ts"] # [B,]
            batch_n=embed_data["batch_n"] # [B,N]
            batch_n_t=embed_data["batch_n_t"] # [B,N] 
            batch_n_ts=embed_data["batch_n_ts"] # [B,N]
            batch_n_mask=embed_data["batch_n_mask"] # [B,N]

            batch_size,max_n=batch_n.size()
            batch_n=batch_n.flatten() # [B,N] -> [B x N,]
            batch_n_t=batch_n_t.flatten() # [B,N] -> [B x N,]
            n_embedding=self.compute_embedding(
                batch_tar=batch_n,
                batch_t=batch_n_t,
                n_layer=n_layer-1
            ) # [B x N,latent_dim]

            ### aggregation
            # time encoding
            batch_tar_ts=batch_tar_ts.unsqueeze(-1) # -> [B,1]
            batch_n_ts=batch_n_ts.unsqueeze(-1) # -> [B,N,1]
            batch_tar_ts_ft=self.time_encoder(batch_tar_ts) # -> [B,time_dim]
            batch_n_ts_ft=self.time_encoder(batch_n_ts) # -> [B,N,time_dim]

            # reshape
            n_embedding=n_embedding.reshape(batch_size,max_n,-1) # -> [B,N,latent_dim]

            # aggregate
            updated_batch_tar_ft=self.aggregate(
                tar_ft=batch_tar_ft,
                tar_ts_ft=batch_tar_ts_ft,
                n_ft=n_embedding,
                n_ts_ft=batch_n_ts_ft,
                n_mask=batch_n_mask,
                n_layer=n_layer
            )
            return updated_batch_tar_ft
    
    def aggregate(self,
            tar_ft:torch.Tensor,
            tar_ts_ft:torch.Tensor,
            n_ft:torch.Tensor,
            n_ts_ft:torch.Tensor,
            n_mask:torch.Tensor,
            n_layer:int):
        """
        """
        aggregation_model=self.attn_layers[n_layer-1]
        output=aggregation_model(
            tar_ft=tar_ft,
            tar_ts_ft=tar_ts_ft,
            n_ft=n_ft,
            n_ts_ft=n_ts_ft,
            n_mask=n_mask
        )
        return output # [B,latent_dim]