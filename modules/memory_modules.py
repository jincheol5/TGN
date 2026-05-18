import torch 
import torch.nn as nn
from typing_extensions import Literal
from .time_encoding import TimeEncoder
from .data import TemporalGraphData 

class MemoryUpdater(nn.Module):
    def __init__(self,
            memory_dim:int,
            time_encoder:TimeEncoder,
            data:TemporalGraphData,
            aggr_fn:Literal["last","mean"]="last"
        ):
        super().__init__()
        self.time_encoder=time_encoder
        self.data=data
        self.aggr_fn=aggr_fn
        self.memory_dim=memory_dim

    def aggregate_memory(self,
            tar_memory,
            tar_t
        ):
        """
        Memory Aggregation
        Input:
            tar_memory: [B,memory_dim]
            tar_t: [B,1]
        Output:
            tar
        """
        match self.aggr_fn:
            case "mean":
                """
                """


    def update_memory_implement(self):
        return NotImplemented

    def update_memory(self):
        self.aggregate_memory()
        return self.update_memory_implement()

class GRUMemoryUpdater(MemoryUpdater):
    def __init__(self,


        ):
        super(GRUMemoryUpdater,self).__init__()
    def update_memory_implement(self):
        """
        """