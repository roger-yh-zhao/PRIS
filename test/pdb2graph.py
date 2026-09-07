import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import argparse
import os, sys
import MDAnalysis as mda
#sys.path.append("/home/shenchao/rtmscorepyg/code")
sys.path.append("..")
from torch_geometric.loader import DataLoader
from PRIS.data.data import VSDataset4test,PDBbindDataset
from PRIS.model_seq.utils import run_an_eval_epoch
from PRIS.model_seq.model import PRIScore, GraphTransformer, GatedGCN
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#you need to set the babel libdir first if you need to generate the pocket
os.environ["BABEL_LIBDIR"] = "/mnt/yihaozhao/anaconda3/envs/pri/lib/openbabel/3.1.0"



def main():

	cut=10
	ids='1zdk'

	for pdbid in ids:


		ligslist = [ "lig%s.pt"%(ids) ] + ["1zdk_n.pdb"   ]
		proslist = [ "pro%s.pt"%(ids) ] + ["1zdk_p.pdb" ]


		data = VSDataset4test(ligs=ligslist,
    					pros=proslist,
    					cutoff=cut,		
    					gen_pocket=False,
    					reflig=None,
    					explicit_H=False, 
    					use_chirality=True,
    					parallel=False);


if __name__ == '__main__':
    main()



