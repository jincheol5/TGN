import torch 
import torch.nn as nn
from typing_extensions import Literal
from .time_encoding import TimeEncoder
from .data import MemoryData 

class MemoryUpdater(nn.Module):
    def __init__(self,
            mem_dim:int,
            msg_dim:int,
            time_dim:int,
            time_encoder:TimeEncoder,
            memory_data:MemoryData,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super().__init__()
        self.memory_data=memory_data
        self.msg_fn=msg_fn
        self.aggr_fn=aggr_fn
        self.mem_dim=mem_dim
        self.msg_dim=msg_dim
        self.time_dim=time_dim

        # set module
        self.time_encoder=time_encoder
        if msg_fn=="mlp":
            self.src_mlp=nn.Sequential(
                nn.Linear(in_features=mem_dim+mem_dim+time_dim,out_features=msg_dim),
                nn.ReLU(),
                nn.Linear(in_features=msg_dim,out_features=msg_dim)
            )
            self.tar_mlp=nn.Sequential(
                nn.Linear(in_features=mem_dim+mem_dim+time_dim,out_features=msg_dim),
                nn.ReLU(),
                nn.Linear(in_features=msg_dim,out_features=msg_dim)
            )

    def create_message(self,
            src,
            tar,
            event_t
        ):
        """
        Input:
            src: [B,]
            tar: [B,]
            event_t: [B,]
        Output:
            src_msg: [B,msg_dim]
            tar_msg: [B,msg_dim]
        """
        src_mem=self.memory_data.get_batch_memory(batch_node=src) # [B,mem_dim]
        src_ts=self.memory_data.get_batch_timespan(batch_node=src,batch_t=event_t) # [B,1]
        src_ts_encoding=self.time_encoder(src_ts) # [B,time_dim]

        tar_mem=self.memory_data.get_batch_memory(batch_node=tar) # [B,mem_dim]
        tar_ts=self.memory_data.get_batch_timespan(batch_node=tar,batch_t=event_t) # [B,1]
        tar_ts_encoding=self.time_encoder(tar_ts) # [B,time_dim]

        src_msg=torch.concat(
            src_mem,
            tar_mem,
            src_ts_encoding,
            dim=-1
        ) # [B,mem_dim+mem_dim+time_dim]
        tar_msg=torch.concat(
            tar_mem,
            src_mem,
            tar_ts_encoding,
            dim=-1
        ) # [B,mem_dim+mem_dim+time_dim]

        if self.msg_fn=="mlp":
            src_msg=self.src_mlp(src_msg) # [B,msg_dim]
            tar_msg=self.tar_mlp(tar_msg) # [B,msg_dim]
        return src_msg,tar_msg

    def aggregate_message(self,
            src,
            tar,
            src_msg,
            tar_msg,
            event_t
        ):
        """
        Memory Aggregation
        Input:
            src: [B,]
            tar: [B,]
            src_msg: [B,msg_dim]
            tar_msg: [B,msg_dim]
            event_t: [B,]
        Output:
            aggregated_node: [unique_N,]
            aggregated_node_msg: [unique_N,msg_dim]
        """
        match self.aggr_fn:
            case "last":
                """
                """
            case "mean":
                """
                """





    def update_memory_implement(self):
        return NotImplemented

    def update_memory(self):
        self.aggregate_message()
        return self.update_memory_implement()

class GRUMemoryUpdater(MemoryUpdater):
    def __init__(self,


        ):
        super(GRUMemoryUpdater,self).__init__()
    def update_memory_implement(self):
        """
        """