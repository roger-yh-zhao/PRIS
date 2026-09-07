import pandas as pd
import numpy as np
import torch as th
from rdkit import Chem
import os
import tempfile
import shutil
from joblib import Parallel, delayed
from torch_geometric.data import Batch, Data, Dataset  #, InMemoryDataset
from ..feats.mol2graph_rdmda_res import prot_to_graph, nuc_to_graph, mol_to_graph, load_mol
from ..feats.extract_pocket_prody import extract_pocket
import MDAnalysis as mda

class PDBbindDataset(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				prots=None,
				labels=None,
				):
		super(PDBbindDataset, self).__init__()
		self.labels = labels
		if isinstance(ids,np.ndarray) or isinstance(ids,list):
			self.pdbids = ids
		else:
			try:
				self.pdbids = np.load(ids)
			except:
				raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
			if self.pdbids.shape[0] == 1:
				pass
			elif self.pdbids.shape[0] == 2:
				self.labels = self.pdbids[-1].astype(float)
				self.pdbids = self.pdbids[0]
			else:
				raise ValueError('the file to store numpy.ndarray should have one/two dimensions')	
		
		if isinstance(ligs,np.ndarray) or isinstance(ligs,tuple) or isinstance(ligs,list):
			if isinstance(ligs[0],Data):
				self.gls = ligs
			else:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) torch_geometric.data.Data objects.')
		else:
			try:
				self.gls = th.load(ligs)
			except:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) torch_geometric.data.Data objects.')
		
		if isinstance(prots,np.ndarray) or isinstance(prots,th.Tensor) or isinstance(prots,list):
			if isinstance(prots[0],Data):
				self.gps = prots
			else:
				raise ValueError('the variable "prots" should be a set of (or a file to store) torch_geometric.data.Data objects.')	
		else:
			try:
				self.gps = th.load(prots)
			except:
				raise ValueError('the variable "prots" should be a set of (or a file to store) torch_geometric.data.Data objects.')	
		
		self.gls = Batch.from_data_list(self.gls)
		self.gps = Batch.from_data_list(self.gps)
		assert len(self.pdbids) == self.gls.num_graphs == self.gps.num_graphs
		if self.labels is None:
			self.labels = th.zeros(len(self.pdbids))
		else:
			self.labels = th.tensor(self.labels)
	
	def len(self):
		return len(self.pdbids)
	
	def get(self, idx):
		pdbid = self.pdbids[idx]
		gp = self.gps[idx]
		gl = self.gls[idx]
		label = self.labels[idx]
		return pdbid, gp, gl, label
	
	def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
		#random.seed(seed)
		np.random.seed(seed)
		if valnum is None:
			valnum = int(valfrac * len(self.pdbids))
		val_inds = np.random.choice(np.arange(len(self.pdbids)),valnum, replace=False)
		train_inds = np.setdiff1d(np.arange(len(self.pdbids)),val_inds)
		return train_inds, val_inds
    
    



class DatasetLLM(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				prots=None,
				labels=None,
                llm=None,
				):
		super(DatasetLLM, self).__init__()
		self.labels = labels
		if isinstance(ids,np.ndarray) or isinstance(ids,list):
			self.pdbids = ids
		else:
			try:
				self.pdbids = np.load(ids)
			except:
				raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
			if self.pdbids.shape[0] == 1:
				pass
			elif self.pdbids.shape[0] == 2:
				self.labels = self.pdbids[-1].astype(float)
				self.pdbids = self.pdbids[0]
			else:
				raise ValueError('the file to store numpy.ndarray should have one/two dimensions')	
		
		if isinstance(ligs,np.ndarray) or isinstance(ligs,tuple) or isinstance(ligs,list):
			if isinstance(ligs[0],Data):
				self.gls = ligs
			else:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) torch_geometric.data.Data objects.')
		else:
			try:
				self.gls = th.load(ligs)
			except:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) torch_geometric.data.Data objects.')
		
		if isinstance(prots,np.ndarray) or isinstance(prots,th.Tensor) or isinstance(prots,list):
			if isinstance(prots[0],Data):
				self.gps = prots
			else:
				raise ValueError('the variable "prots" should be a set of (or a file to store) torch_geometric.data.Data objects.')	
		else:
			try:
				self.gps = th.load(prots)
			except:
				raise ValueError('the variable "prots" should be a set of (or a file to store) torch_geometric.data.Data objects.')	
		
		self.gls = Batch.from_data_list(self.gls)
		self.gps = Batch.from_data_list(self.gps)
		assert len(self.pdbids) == self.gls.num_graphs == self.gps.num_graphs
		if self.labels is None:
			self.labels = th.zeros(len(self.pdbids))
		else:
			self.labels = th.tensor(self.labels)
            
		#print(self.gps, llm) 
		if type(llm) == str and llm[-3:] == '.pt':
			esm3emb = th.load(llm)
			esm3emb = th.cat(esm3emb, dim=0)#;print(esm3emb.shape)
			#esm3emb = esm3emb[:832321]#;print(esm3emb.shape);print(esm3emb)
			#print(self.gps.x.shape)#;print(self.gps.x)	
			esm3emb = esm3emb.to(self.gps.x.device)	
			self.gps.x = th.cat([self.gps.x, esm3emb], dim=1)
			#print(self.gps.x.shape);print(self.gps.x)


	def len(self):
		return len(self.pdbids)
	
	def get(self, idx):
		pdbid = self.pdbids[idx]
		gp = self.gps[idx]
		gl = self.gls[idx]
		label = self.labels[idx]
		return pdbid, gp, gl, label
	
	def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
		#random.seed(seed)
		np.random.seed(seed)
		if valnum is None:
			valnum = int(valfrac * len(self.pdbids))
		val_inds = np.random.choice(np.arange(len(self.pdbids)),valnum, replace=False)
		train_inds = np.setdiff1d(np.arange(len(self.pdbids)),val_inds)
		return train_inds, val_inds
    
    
    
    

	

class VSDataset(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				prot=None,
				labels=None,
				gen_pocket=False,
				cutoff=None,
				reflig=None,
				explicit_H=False, 
				use_chirality=True,
				parallel=True			
				):
		super(VSDataset, self).__init__()
		self.labels = labels
		self.gp=None
		self.gls=None
		self.pocketdir = None
		self.prot = None
		self.ligs = None
		self.cutoff = cutoff
		self.explicit_H=explicit_H
		self.use_chirality=use_chirality
		self.parallel=parallel
		
		if isinstance(prot, Chem.rdchem.Mol):
			assert gen_pocket == False
			self.prot = prot
			self.gp = prot_to_graph(self.prot, cutoff)
		else:
			if gen_pocket:
				if cutoff is None or reflig is None:
					raise ValueError('If you want to generate the pocket, the cutoff and the reflig should be given')
				try:
					self.pocketdir = tempfile.mkdtemp()
					extract_pocket(prot, reflig, cutoff, 
								protname="temp",
								workdir=self.pocketdir)
					pocket = load_mol("%s/temp_pocket_%s.pdb"%(self.pocketdir, cutoff), 
								explicit_H=explicit_H, use_chirality=use_chirality)
					self.prot = pocket
					self.gp = prot_to_graph(self.prot, cutoff)
				except:
					raise ValueError('The graph of pocket cannot be generated')
			else:
				try:
					###pocket = load_mol(prot, explicit_H=explicit_H, use_chirality=use_chirality)
					#self.graphp = mol_to_graph(pocket, explicit_H=explicit_H, use_chirality=use_chirality)	
					###self.prot = pocket
					#self.gp = prot_to_graph(self.prot, cutoff)
					self.gp = prot_to_graph(prot, cutoff)#;print('see',self.gp)
				except:
					print('wrong input, need to check the file', prot) #skip
					self.gp = [nuc_to_graph(ligs, cutoff)]  #skip
					#raise ValueError('The graph of pocket cannot be generated')
			
		if isinstance(ligs,np.ndarray) or isinstance(ligs,list):
			if isinstance(ligs[0], Chem.rdchem.Mol):
				self.ligs = ligs
				self.gls = self._mol_to_graph()
			elif isinstance(ligs[0], Data):
				self.gls = ligs
			else:
				raise ValueError('Ligands should be a list of rdkit.Chem.rdchem.Mol objects')
		else:
			if ligs.endswith(".mol2"):
				lig_blocks = self._mol2_split(ligs)	
				self.ligs = [Chem.MolFromMol2Block(lig_block) for lig_block in lig_blocks]
				self.gls = self._mol_to_graph()
			elif ligs.endswith(".sdf"):
				lig_blocks = self._sdf_split(ligs)	
				self.ligs = [Chem.MolFromMolBlock(lig_block) for lig_block in lig_blocks]
				self.gls = self._mol_to_graph()
			elif ligs.endswith(".pdb"):
				try:
					self.ligs = Chem.MolFromPDBFile(ligs)
					#self.gls = [nuc_to_graph(prot.ligs, cutoff)]
					self.gls = [nuc_to_graph(ligs, cutoff)]

				except:
					print('wrong input, need to check the file', ligs) #skip
					self.gls = [self.gp] #skip   				
				self.ligs = [self.ligs]
                
			else:
				try:	
					self.gls,_ = load_graphs(ligs)
				except:
					raise ValueError('Only the ligands with .sdf or .mol2 or a file to genrate DGLGraphs will be supported')
		
		if ids is None:
			if self.ligs is not None:
				self.idsx = ["%s-%s"%(self.get_ligname(lig),i) for i, lig in enumerate(self.ligs)]
			else:
				self.idsx = ["lig%s"%i for i in range(len(self.gls))]
		else:
			self.idsx = ids
		
		self.ids, self.gls = zip(*filter(lambda x: x[1] != None, zip(self.idsx, self.gls)))
		self.ids = list(self.ids)
		self.gls = Batch.from_data_list(self.gls)
		assert len(self.ids) == self.gls.num_graphs
		if self.labels is None:
			self.labels = th.zeros(len(self.ids))
		else:
			self.labels = th.tensor(self.labels)
		
		if self.pocketdir is not None:
			shutil.rmtree(self.pocketdir)

	def len(self):
		return len(self.ids)
	
	def get(self, idx):
		id = self.ids[idx]
		gp = self.gp
		gl = self.gls[idx]
		label = self.labels[idx] 
		return id, gp, gl, label	
		
	def _mol2_split(self, infile):
		contents = open(infile, 'r').read()
		return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]
	
	def _sdf_split(self, infile):
		contents = open(infile, 'r').read()
		return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]
	
	def _mol_to_graph0(self, lig):
		try:
			gx = mol_to_graph(lig, explicit_H=self.explicit_H, use_chirality=self.use_chirality)
		except:
			print("failed to scoring for {} and {}".format(self.gp, lig))
			return None
		return gx

	def _mol_to_graph(self):
		if self.parallel:
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._mol_to_graph0)(lig) for lig in self.ligs)
		else:
			graphs = []
			for lig in self.ligs:
				graphs.append(self._mol_to_graph0(lig))
			return graphs
	
	def get_ligname(self, m):
		if m is None:
			return None
		else:
			if m.HasProp("_Name"):
				return m.GetProp("_Name")
			else:
				return None

class VSDataset4test(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				pros=None,
				labels=None,
				gen_pocket=False,
				cutoff=None,
				reflig=None,
				explicit_H=False, 
				use_chirality=True,
				parallel=True			
				):
		super(VSDataset4test, self).__init__()
		self.labels = labels
		self.gps=None
		self.gls=None
		self.pocketdir = None
		self.prot = None
		self.ligs = None
		self.cutoff = cutoff
		self.explicit_H=explicit_H
		self.use_chirality=use_chirality
		self.parallel=parallel
		
		if isinstance(pros, Chem.rdchem.Mol):
			assert gen_pocket == False
			self.pros = pros
			self.gps = prot_to_graph(self.prot, cutoff)
		else:
			if gen_pocket:
				if cutoff is None or reflig is None:
					raise ValueError('If you want to generate the pocket, the cutoff and the reflig should be given')
				try:
					self.pocketdir = tempfile.mkdtemp()
					extract_pocket(pros, reflig, cutoff, 
								protname="temp",
								workdir=self.pocketdir)
					pocket = load_mol("%s/temp_pocket_%s.pdb"%(self.pocketdir, cutoff), 
								explicit_H=explicit_H, use_chirality=use_chirality)
					self.prot = pocket
					self.gps = prot_to_graph(self.prot, cutoff)
				except:
					raise ValueError('The graph of pocket cannot be generated')
			else:
				try:
					###pocket = load_mol(prot, explicit_H=explicit_H, use_chirality=use_chirality)
					#self.graphp = mol_to_graph(pocket, explicit_H=explicit_H, use_chirality=use_chirality)	
					###self.prot = pocket
					#self.gp = prot_to_graph(self.prot, cutoff)
					self.gps = self._prot_to_graph(pros, cutoff)#;print('see',self.gps)
				except:
					print('wrong input, need to check the file', pros) #skip
					#self.gps = self._prot_to_graph(pros, cutoff)#4check
					#self.gp = [nuc_to_graph(ligs, cutoff)]  #skip
					#raise ValueError('The graph of pocket cannot be generated')

		th.save(self.gps, pros[0] )
			
		if isinstance(ligs,np.ndarray): #or isinstance(ligs,list):
			if isinstance(ligs[0], Chem.rdchem.Mol):
				self.ligs = ligs
				self.gls = self._mol_to_graph()
			elif isinstance(ligs[0], Data):
				self.gls = ligs
			else:
				raise ValueError('Ligands should be a list of rdkit.Chem.rdchem.Mol objects')
		else:
			if ligs[-1].endswith(".mol2"):
				lig_blocks = self._mol2_split(ligs)	
				self.ligs = [Chem.MolFromMol2Block(lig_block) for lig_block in lig_blocks]
				self.gls = self._mol_to_graph()
			elif ligs[-1].endswith(".sdf"):
				lig_blocks = self._sdf_split(ligs)	
				self.ligs = [Chem.MolFromMolBlock(lig_block) for lig_block in lig_blocks]
				self.gls = self._mol_to_graph()
			elif ligs[-1].endswith(".pdb"):
				try:
					#self.ligs = Chem.MolFromPDBFile(ligs)
					#self.gls = [nuc_to_graph(prot.ligs, cutoff)]
					#self.gls = [nuc_to_graph(ligs, cutoff)]
					self.gls = self._nuc_to_graph(ligs, cutoff)
                    
				except:
					print('wrong input, need to check the file', ligs) #skip
					#self.gls = [self.gp] #skip   				
				#self.ligs = [self.ligs]
                
			else:
				try:	
					self.gls,_ = load_graphs(ligs)
				except:
					raise ValueError('Only the ligands with .sdf or .mol2 or a file to genrate DGLGraphs will be supported')
                    
		th.save(self.gls, ligs[0] )
		
		'''
		if ids is None:
			if self.ligs is not None:
				self.idsx = ["%s-%s"%(self.get_ligname(lig),i) for i, lig in enumerate(self.ligs)]
			else:
				self.idsx = ["lig%s"%i for i in range(len(self.gls))]
		else:
			self.idsx = ids
		
		self.ids, self.gls = zip(*filter(lambda x: x[1] != None, zip(self.idsx, self.gls)))
		self.ids = list(self.ids)
		'''
		#self.gls = Batch.from_data_list(self.gls)
		'''
		assert len(self.ids) == self.gls.num_graphs
		if self.labels is None:
			self.labels = th.zeros(len(self.ids))
		else:
			self.labels = th.tensor(self.labels)
		
		if self.pocketdir is not None:
			shutil.rmtree(self.pocketdir)
		'''

	def len(self):
		return len(self.ids)
	
	def get(self, idx):
		id = self.ids[idx]
		gps = self.gps
		gl = self.gls[idx]
		label = self.labels[idx] 
		return id, gps, gl, label	
		
	def _mol2_split(self, infile):
		contents = open(infile, 'r').read()
		return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]
	
	def _sdf_split(self, infile):
		contents = open(infile, 'r').read()
		return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]
	
	def _mol_to_graph0(self, lig):
		try:
			gx = mol_to_graph(lig, explicit_H=self.explicit_H, use_chirality=self.use_chirality)
		except:
			print("failed to scoring for {} and {}".format(self.gps, lig))
			return None
		return gx

	def _mol_to_graph(self):
		if self.parallel:
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._mol_to_graph0)(lig) for lig in self.ligs)
		else:
			graphs = []
			for lig in self.ligs:
				graphs.append(self._mol_to_graph0(lig))
			return graphs
        
	def _nuc_to_graph0(self, lig,cutoff):
		print("{}".format(lig),flush=True)
		try:
			gx = nuc_to_graph(lig, cutoff)
		except:
			print("failed to scoring for {}".format(lig))
			return None
		return gx
        
	def _nuc_to_graph(self,ligs,cutoff):
		if self.parallel:
			#undone
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._nuc_to_graph0)(lig,cutoff) for lig in ligs[1:])
		else:
			graphs = []
			for lig in ligs[1:]:
				graphs.append(self._nuc_to_graph0(lig,cutoff))

			return graphs
        
	def _prot_to_graph0(self, pro, cutoff):
		print("{}".format(pro),flush=True)
		try:
			gx = prot_to_graph(pro, cutoff)
		except:
			print("failed to scoring for {}".format(pro))
			#gx = prot_to_graph(pro, cutoff)#4check
			return None
		return gx
        
	def _prot_to_graph(self,pros, cutoff):
		print(self.parallel)
		if self.parallel:
			#undone
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._prot_to_graph0)(pro,cutoff) for pro in pros[1:])
		else:
			graphs = []
			for pro in pros[1:]:
				graphs.append(self._prot_to_graph0(pro,cutoff))
				
			return graphs

	
	def get_ligname(self, m):
		if m is None:
			return None
		else:
			if m.HasProp("_Name"):
				return m.GetProp("_Name")
			else:
				return None
            

class PDB_convert(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				pros=None,
				labels=None,
				gen_pocket=False,
				cutoff=10,
				reflig=None,
				explicit_H=False, 
				use_chirality=True,
				parallel=True			
				):
		super(PDB_convert, self).__init__()
		self.labels = labels
		self.gps=None
		self.gls=None
		self.pocketdir = None
		self.prot = None
		self.ligs = None
		self.cutoff = cutoff
		self.explicit_H=explicit_H
		self.use_chirality=use_chirality
		self.parallel=parallel
		
		if isinstance(pros, Chem.rdchem.Mol):
			assert gen_pocket == False
			self.pros = pros
			self.gps = prot_to_graph(self.prot, cutoff)
		else:
			if gen_pocket:
				if cutoff is None or reflig is None:
					raise ValueError('If you want to generate the pocket, the cutoff and the reflig should be given')
				try:
					self.pocketdir = tempfile.mkdtemp()
					extract_pocket(pros, reflig, cutoff, 
								protname="temp",
								workdir=self.pocketdir)
					pocket = load_mol("%s/temp_pocket_%s.pdb"%(self.pocketdir, cutoff), 
								explicit_H=explicit_H, use_chirality=use_chirality)
					self.prot = pocket
					self.gps = prot_to_graph(self.prot, cutoff)
				except:
					raise ValueError('The graph of pocket cannot be generated')
			else:
				try:
					###pocket = load_mol(prot, explicit_H=explicit_H, use_chirality=use_chirality)
					#self.graphp = mol_to_graph(pocket, explicit_H=explicit_H, use_chirality=use_chirality)	
					###self.prot = pocket
					#self.gp = prot_to_graph(self.prot, cutoff)
					self.gps = self._prot_to_graph(pros, cutoff)#;print('see',self.gps)
				except:
					print('wrong input, need to check the file', pros) #skip
					#self.gp = [nuc_to_graph(ligs, cutoff)]  #skip
					#raise ValueError('The graph of pocket cannot be generated')

		th.save(self.gps, pros[0] )
			
		if isinstance(ligs,np.ndarray): #or isinstance(ligs,list):
			if isinstance(ligs[0], Chem.rdchem.Mol):
				self.ligs = ligs
				self.gls = self._mol_to_graph()
			elif isinstance(ligs[0], Data):
				self.gls = ligs
			else:
				raise ValueError('Ligands should be a list of rdkit.Chem.rdchem.Mol objects')
		else:
			if ligs[-1].endswith(".mol2"):
				lig_blocks = self._mol2_split(ligs)	
				self.ligs = [Chem.MolFromMol2Block(lig_block) for lig_block in lig_blocks]
				self.gls = self._mol_to_graph()
			elif ligs[-1].endswith(".sdf"):
				lig_blocks = self._sdf_split(ligs)	
				self.ligs = [Chem.MolFromMolBlock(lig_block) for lig_block in lig_blocks]
				self.gls = self._mol_to_graph()
			elif ligs[-1].endswith(".pdb"):
				try:
					#self.ligs = Chem.MolFromPDBFile(ligs)
					#self.gls = [nuc_to_graph(prot.ligs, cutoff)]
					#self.gls = [nuc_to_graph(ligs, cutoff)]
					self.gls = self._nuc_to_graph(ligs, cutoff)
                    
				except:
					print('wrong input, need to check the file', ligs) #skip
					#self.gls = [self.gp] #skip   				
				#self.ligs = [self.ligs]
                
			else:
				try:	
					self.gls,_ = load_graphs(ligs)
				except:
					raise ValueError('Only the ligands with .sdf or .mol2 or a file to genrate DGLGraphs will be supported')
                    
		th.save(self.gls, ligs[0] )
		
		'''
		if ids is None:
			if self.ligs is not None:
				self.idsx = ["%s-%s"%(self.get_ligname(lig),i) for i, lig in enumerate(self.ligs)]
			else:
				self.idsx = ["lig%s"%i for i in range(len(self.gls))]
		else:
			self.idsx = ids
		
		self.ids, self.gls = zip(*filter(lambda x: x[1] != None, zip(self.idsx, self.gls)))
		self.ids = list(self.ids)
		'''
		#self.gls = Batch.from_data_list(self.gls)
		'''
		assert len(self.ids) == self.gls.num_graphs
		if self.labels is None:
			self.labels = th.zeros(len(self.ids))
		else:
			self.labels = th.tensor(self.labels)
		
		if self.pocketdir is not None:
			shutil.rmtree(self.pocketdir)
		'''

	def len(self):
		return len(self.ids)
	
	def get(self, idx):
		id = self.ids[idx]
		gps = self.gps
		gl = self.gls[idx]
		label = self.labels[idx] 
		return id, gps, gl, label	
		
	def _mol2_split(self, infile):
		contents = open(infile, 'r').read()
		return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]
	
	def _sdf_split(self, infile):
		contents = open(infile, 'r').read()
		return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]
	
	def _mol_to_graph0(self, lig):
		try:
			gx = mol_to_graph(lig, explicit_H=self.explicit_H, use_chirality=self.use_chirality)
		except:
			print("failed to scoring for {} and {}".format(self.gps, lig))
			return None
		return gx

	def _mol_to_graph(self):
		if self.parallel:
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._mol_to_graph0)(lig) for lig in self.ligs)
		else:
			graphs = []
			for lig in self.ligs:
				graphs.append(self._mol_to_graph0(lig))
			return graphs
        
	def _nuc_to_graph0(self, lig,cutoff):
		print("{}".format(lig),flush=True)
		try:
			gx = nuc_to_graph(lig, cutoff)
		except:
			print("failed to scoring for {}".format(lig))
			return None
		return gx
        
	def _nuc_to_graph(self,ligs,cutoff):
		if self.parallel:
			#undone
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._nuc_to_graph0)(lig,cutoff) for lig in ligs[1:])
		else:
			graphs = []
			for lig in ligs[1:]:
				graphs.append(self._nuc_to_graph0(lig,cutoff))

			return graphs
        
	def _prot_to_graph0(self, pro, cutoff):
		print("{}".format(pro),flush=True)
		try:
			gx = prot_to_graph(pro, cutoff)
		except:
			print("failed to scoring for {}".format(pro))
			return None
		return gx
        
	def _prot_to_graph(self,pros, cutoff):
		print(self.parallel)
		if self.parallel:
			#undone
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._prot_to_graph0)(pro,cutoff) for pro in pros[1:])
		else:
			graphs = []
			for pro in pros[1:]:
				graphs.append(self._prot_to_graph0(pro,cutoff))
				
			return graphs

	
	def get_ligname(self, m):
		if m is None:
			return None
		else:
			if m.HasProp("_Name"):
				return m.GetProp("_Name")
			else:
				return None

	
class PNDataset(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				prot=None,
				labels=None,
				gen_pocket=False,
				cutoff=None,
				reflig=None,
				explicit_H=False, 
				use_chirality=True,
				parallel=True			
				):
		super(PNDataset, self).__init__()
		self.labels = labels
		self.gp=None
		self.gls=None
		self.pocketdir = None
		self.prot = None
		self.ligs = None
		self.cutoff = cutoff
		self.explicit_H=explicit_H
		self.use_chirality=use_chirality
		self.parallel=parallel
		
		if isinstance(prot, Chem.rdchem.Mol):
			assert gen_pocket == False
			self.prot = prot
			self.gp = prot_to_graph(self.prot, cutoff)
		else:
			if gen_pocket:
				if cutoff is None or reflig is None:
					raise ValueError('If you want to generate the pocket, the cutoff and the reflig should be given')
				try:
					self.pocketdir = tempfile.mkdtemp()
					extract_pocket(prot, reflig, cutoff, 
								protname="temp",
								workdir=self.pocketdir)
					pocket = load_mol("%s/temp_pocket_%s.pdb"%(self.pocketdir, cutoff), 
								explicit_H=explicit_H, use_chirality=use_chirality)
					self.prot = pocket
					self.gp = prot_to_graph(self.prot, cutoff)
				except:
					raise ValueError('The graph of pocket cannot be generated')
			else:
				try:
					###pocket = load_mol(prot, explicit_H=explicit_H, use_chirality=use_chirality)
					#self.graphp = mol_to_graph(pocket, explicit_H=explicit_H, use_chirality=use_chirality)	
					###self.prot = pocket
					#self.gp = prot_to_graph(self.prot, cutoff)
					self.gp = prot_to_graph(prot, cutoff)
				except:
					print('wrong input, need to check the file', prot) #skip
					self.gp = [nuc_to_graph(ligs, cutoff)]  #skip
					#raise ValueError('The graph of pocket cannot be generated')
			
		if isinstance(ligs,np.ndarray) or isinstance(ligs,list):
			if isinstance(ligs[0], Chem.rdchem.Mol):
				self.ligs = ligs
				self.gls = self._mol_to_graph()
			elif isinstance(ligs[0], Data):
				self.gls = ligs
			else:
				raise ValueError('Ligands should be a list of rdkit.Chem.rdchem.Mol objects')
		else:
			if ligs.endswith(".mol2"):
				lig_blocks = self._mol2_split(ligs)	
				self.ligs = [Chem.MolFromMol2Block(lig_block) for lig_block in lig_blocks]
				self.gls = self._mol_to_graph()
			elif ligs.endswith(".sdf"):
				lig_blocks = self._sdf_split(ligs)	
				self.ligs = [Chem.MolFromMolBlock(lig_block) for lig_block in lig_blocks]
				self.gls = self._mol_to_graph()
			elif ligs.endswith(".pdb"):
				try:
					self.ligs = Chem.MolFromPDBFile(ligs)
					#self.gls = [nuc_to_graph(prot.ligs, cutoff)]
					self.gls = [nuc_to_graph(ligs, cutoff)]

				except:
					print('wrong input, need to check the file', ligs) #skip
					self.gls = [self.gp] #skip   				
				self.ligs = [self.ligs]
                
			else:
				try:	
					self.gls,_ = load_graphs(ligs)
				except:
					raise ValueError('Only the ligands with .sdf or .mol2 or a file to genrate DGLGraphs will be supported')
		
		if ids is None:
			if self.ligs is not None:
				self.idsx = ["%s-%s"%(self.get_ligname(lig),i) for i, lig in enumerate(self.ligs)]
			else:
				self.idsx = ["lig%s"%i for i in range(len(self.gls))]
		else:
			self.idsx = ids
		
		self.ids, self.gls = zip(*filter(lambda x: x[1] != None, zip(self.idsx, self.gls)))
		self.ids = list(self.ids)
		self.gls = Batch.from_data_list(self.gls)
		assert len(self.ids) == self.gls.num_graphs
		if self.labels is None:
			self.labels = th.zeros(len(self.ids))
		else:
			self.labels = th.tensor(self.labels)
		
		if self.pocketdir is not None:
			shutil.rmtree(self.pocketdir)

	def len(self):
		return len(self.ids)
	
	def get(self, idx):
		id = self.ids[idx]
		gp = self.gp
		gl = self.gls[idx]
		label = self.labels[idx] 
		return id, gp, gl, label	
		
	def _mol2_split(self, infile):
		contents = open(infile, 'r').read()
		return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]
	
	def _sdf_split(self, infile):
		contents = open(infile, 'r').read()
		return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]
	
	def _mol_to_graph0(self, lig):
		try:
			gx = mol_to_graph(lig, explicit_H=self.explicit_H, use_chirality=self.use_chirality)
		except:
			print("failed to scoring for {} and {}".format(self.gp, lig))
			return None
		return gx

	def _mol_to_graph(self):
		if self.parallel:
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._mol_to_graph0)(lig) for lig in self.ligs)
		else:
			graphs = []
			for lig in self.ligs:
				graphs.append(self._mol_to_graph0(lig))
			return graphs
	
	def get_ligname(self, m):
		if m is None:
			return None
		else:
			if m.HasProp("_Name"):
				return m.GetProp("_Name")
			else:
				return None