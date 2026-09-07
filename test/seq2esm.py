from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig, LogitsConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

import os
import torch
os.environ["INFRA_PROVIDER"] = "True"
device = torch.device("cuda:0")
model = ESM3.from_pretrained('esm3_sm_open_v1',device=device)



def get_emb(inp):
    embeddings=[]
    
    s = open(inp).readlines()
    s = [ i[:-1] for i in s ]


    for idx in range(len(s)):
        i = s[idx]
        try:
            seqs = i.split('""')
            
            embedding=[]
            
            for seq in seqs:

                protein = ESMProtein(sequence=seq)
                protein_tensor = model.encode(protein)

                output = model.forward_and_sample(
                    protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
                )
                
                embedding.append( output.per_residue_embedding[1:-1] )

            
            embeddings.append( torch.cat(embedding, dim=0) )

        except:
            print(idx, len(i))

    
    #print(embeddings)
    torch.save(embeddings, inp.replace('.seq','.pt'))

get_emb('esm1zdk.seq')








