import argparse
import torch
from modules import TemporalGraphData,MemoryData
from modules import TimeEncoder,GraphAttentionEmbedding,GRUMemoryUpdater

"""
<< Test >> 
memory_modules.GraphAttentionEmbedding
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. GRUMemoryUpdater.update_memory
            """
            # parameter
            node_dim=4
            mem_dim=4
            msg_dim=4
            latent_dim=4
            time_dim=4
            n_head=4
            n_layer=3
            msg_fn="mlp"
            aggr_fn="last"
            is_memory=True
            batch_size=4

            # data
            eventstream=[
                (3,7,1),
                (3,6,2),
                (1,4,3),
                (1,3,4),

                (1,2,5),
                (2,7,5),
                (1,4,7),
                (4,7,7),

                (4,5,8),
                (1,3,9)
            ]
            graph_data=TemporalGraphData(node_dim=node_dim)
            memory_data=MemoryData(mem_dim=mem_dim)
            time_encoder=TimeEncoder(time_dim=time_dim)
            memory_updater=GRUMemoryUpdater(
                mem_dim=mem_dim,
                msg_dim=msg_dim,
                time_dim=time_dim,
                time_encoder=time_encoder,
                memory_data=memory_data,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )

            graph_embed=GraphAttentionEmbedding(
                node_dim=node_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                graph_data=graph_data,
                memory_data=memory_data,
                time_encoder=time_encoder,
                n_head=n_head,
                n_layer=n_layer,
                is_memory=is_memory
            )
            
            for idx in range(0,len(eventstream),batch_size):
                batch_events=eventstream[idx:idx+batch_size]

                # update data
                graph_data.update_graph(batch_events=batch_events)
                memory_data.update_memory_data(batch_events=batch_events)

                ### execute memory, embed module
                src,tar,event_t=zip(*batch_events)
                src=torch.tensor(src,dtype=torch.long) # [B,]
                tar=torch.tensor(tar,dtype=torch.long) # [B,]
                event_t=torch.tensor(event_t,dtype=torch.float32) # [B,]

                # 1. execute memory module
                memory_updater.update_memory(
                    src=src,
                    tar=tar,
                    event_t=event_t
                )

                # 2. execute embed module
                updated_batch_tar_ft=graph_embed.compute_embedding(
                    batch_tar=tar,
                    batch_t=event_t,
                    n_layer=3
                )

                print(f"updated batch_tar_feature:")
                print(updated_batch_tar_ft)




if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        'test_num':args.test_num
    }
    test_fn(**test_config)