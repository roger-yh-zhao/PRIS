###non-standard residue
import pandas as pd
import numpy as np
from rdkit import Chem
import torch as th
import re, os
from itertools import permutations
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from joblib import Parallel, delayed
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances
import openbabel as ob

import random
import string
import shutil

def generate_random_folder_name(length=8):
    
	random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
	return random_string

METAL = ["LI","NA","K","RB","CS","MG","TL","CU","AG","BE","NI","PT","ZN","CO","PD","AG","CR","FE","V","MN","HG",'GA', 
		"CD","YB","CA","SN","PB","EU","SR","SM","BA","RA","AL","IN","TL","Y","LA","CE","PR","ND","GD","TB","DY","ER",
		"TM","LU","HF","ZR","CE","PU","TH"] 
RES_MAX_NATOMS=24
RES_MAX_NATOMS_NUC=24



def prot_to_graph(prot, cutoff):
	"""obtain the residue graphs"""
	try:
		u = mda.Universe(Chem.MolFromPDBFile(prot))
	except:
		folder_path = generate_random_folder_name()
		os.mkdir(folder_path)
		u = mda.Universe(prot)#;print(prot)
		no_h = u.select_atoms('not element H')
		no_h.write( './%s/1.pdb'%(folder_path) )
		u = mda.Universe( './%s/1.pdb'%(folder_path) )
		shutil.rmtree(folder_path)

		#u = mda.Universe(prot)#;print(prot)
		#no_h = u.select_atoms('not element H')
		#no_h.write( '%spro.pdb'%(prot[28:38]) )
		#u = mda.Universe( '%spro.pdb'%(prot[28:38]) )
		#os.remove( '%spro.pdb'%(prot[28:38]) )
        
	pdb_file = prot
	mol = ob.OBMol()
	obConversion = ob.OBConversion()
	obConversion.SetInFormat("pdb")
	obConversion.ReadFile(mol, pdb_file)
                    
	#mol.AddHydrogens()
	mol.ConnectTheDots()
	mol.PerceiveBondOrders()
        
	# Add nodes
	num_residues = len(u.residues)#;print('resall', u.residues)
	res_feats = np.array([calc_res_features(res) for res in u.residues])#;print(res_feats[:,37:])
	edgeids, distm = obatin_edge(u, cutoff)
	src_list, dst_list = zip(*edgeids)
    
	ca_pos = th.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))
    
	u_backbone = u.atoms.select_atoms("name N or name CA or name C or name O or name CB")
	center_pos = th.tensor(u_backbone.atoms.center_of_mass(compound='residues'))#remove base atoms
	dis_matx_ca = distance_matrix(ca_pos, ca_pos)
	cadist = th.tensor([dis_matx_ca[i,j] for i,j in edgeids]) * 0.1
	dis_matx_center = distance_matrix(center_pos, center_pos)
	cedist = th.tensor([dis_matx_center[i,j] for i,j in edgeids]) * 0.1
    
	edge_connect =  th.tensor(np.array([check_connect_nuc(mol, x, y) for x,y in edgeids]))

	try:
		edge_connect_check =  th.tensor(np.array([check_connect(u, x, y) for x,y in edgeids]))
		if th.equal(edge_connect_check,edge_connect):
			pass
		else:
			print('wrong edges, need to check input file',prot) 
	except:
		pass
    
	edge_feats = th.cat([edge_connect.view(-1,1), cadist.view(-1,1), cedist.view(-1,1), th.tensor(distm)], dim=1)  

	#res_max_natoms = max([len(res.atoms) for res in u.residues])#; print(res_max_natoms)    
	#res_coods = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) for res in u.residues]))
    ###
	#res_coods = th.tensor( np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) if len(res.atoms)<24 else \
    #                             np.concatenate([res.atoms.positions[:24], np.full((RES_MAX_NATOMS-24, 3), np.nan)],axis=0) for res in u.residues]) )#;print(res_coods[0])
      
    #res_coods[:4] is backbone atoms;res_coods[4:] is base atoms         
	res_coods = th.tensor(np.array([np.concatenate([obtain_atom_pos(res,"name N"), obtain_atom_pos(res,"name CA"), obtain_atom_pos(res,"name C"), obtain_atom_pos(res,"name O")\
                                                 , obtain_atom_pos(res,"name CB")\
                                                         ,res.atoms.select_atoms("not (name N or name CA or name C or name O or name CB)").positions\
                                                             , np.full((RES_MAX_NATOMS-5-len( res.atoms.select_atoms("not (name N or name CA or name C or name O or name CB)") ), 3), np.nan)],axis=0) for res in u.residues] ))

    ###
	g = Data(x=th.tensor(res_feats, dtype=th.float), 
			edge_index=th.tensor([src_list, dst_list], dtype=th.int64), 
			pos=res_coods,
			edge_attr=th.tensor(np.array(edge_feats), dtype=th.float))#;print(res_coods.size())
	
	#g.ndata.pop("ca_pos")
	#g.ndata.pop("center_pos")
	#g.ndata["pos"] = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) for res in u.residues]))
	return g


def nuc_to_graph(prot, cutoff):
	"""obtain the residue graphs"""
	try:
		u = mda.Universe(Chem.MolFromPDBFile(prot)) 
	except:
		folder_path = generate_random_folder_name()
		os.mkdir(folder_path)
		u = mda.Universe(prot)#;print(prot)
		no_h = u.select_atoms('not element H')
		no_h.write( './%s/1.pdb'%(folder_path) )
		u = mda.Universe( './%s/1.pdb'%(folder_path) )
		shutil.rmtree(folder_path)
        
	pdb_file = prot
	mol = ob.OBMol()
	obConversion = ob.OBConversion()
	obConversion.SetInFormat("pdb")
	obConversion.ReadFile(mol, pdb_file)
                    
	#mol.AddHydrogens()
	mol.ConnectTheDots()
	mol.PerceiveBondOrders()
        
	# Add nodes
	num_residues = len(u.residues)#;print('resall', u.residues)

	if (num_residues>200):
	    print('big',prot)
	
	res_feats = np.array([calc_res_features_nuc(res_index, u.residues) for res_index in u.residues.resids])#;print(u.residues.resids);quit()
	edgeids, distm = obatin_edge_nuc(u, cutoff)
	try:
		src_list, dst_list = zip(*edgeids)
	except:  ##some residues loss the CA atoms
		src_list=(); dst_list=()#; print('only one')
	
	ca_pos = th.tensor(np.array([obtain_ca_pos_nuc(res) for res in u.residues]))

	u_backbone = u.atoms.select_atoms("name P or name OP1 or name OP2 or name O5' or name C5' or name C4' or name O4' or name C3' or name O3' or name C2' or name O2' or name C1'")
	center_pos = th.tensor(u_backbone.atoms.center_of_mass(compound='residues'))#remove base atoms
	dis_matx_ca = distance_matrix(ca_pos, ca_pos)
	cadist = th.tensor([dis_matx_ca[i,j] for i,j in edgeids]) * 0.1
	dis_matx_center = distance_matrix(center_pos, center_pos)
	cedist = th.tensor([dis_matx_center[i,j] for i,j in edgeids]) * 0.1
    
	edge_connect =  th.tensor(np.array([check_connect_nuc(mol, x, y) for x,y in edgeids]))
    
	try:
		edge_connect_check =  th.tensor(np.array([check_connect(u, x, y) for x,y in edgeids]))
		if th.equal(edge_connect_check,edge_connect):
			pass
		else:
			print('wrong edges, need to check input file',prot) 
	except:
		pass
    
	edge_feats = th.cat([edge_connect.view(-1,1), cadist.view(-1,1), cedist.view(-1,1), th.tensor(distm)], dim=1)#;print([len(res.atoms) for res in u.residues])
    
	#res_max_natoms = max([len(res.atoms) for res in u.residues])#; print(res_max_natoms)    
	#res_coods = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) for res in u.residues]))
	###
	#res_coods = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) if len(res.atoms)<24 else \
    #                             np.concatenate([res.atoms.positions[:24], np.full((RES_MAX_NATOMS-24, 3), np.nan)],axis=0) for res in u.residues]))
    
    #res_coods[:12] is backbone atoms;res_coods[12:] is base atoms    
	res_coods = th.tensor(np.array([np.concatenate([obtain_atom_pos(res,"name P"), obtain_atom_pos(res,"name OP1"), obtain_atom_pos(res,"name OP2"), obtain_atom_pos(res,"name O5'")\
                                                 , obtain_atom_pos(res,"name C5'"), obtain_atom_pos(res,"name C4'"), obtain_atom_pos(res,"name O4'"), obtain_atom_pos(res,"name C3'")\
                                                     , obtain_atom_pos(res,"name O3'"), obtain_atom_pos(res,"name C2'"), obtain_atom_pos(res,"name O2'"), obtain_atom_pos(res,"name C1'")\
                                                         ,res.atoms.select_atoms("not (name P or name OP1 or name OP2 or name O5' or name C5' or name C4' or name O4' or name C3' or name O3' or name C2' or name O2' or name C1')").positions\
                                                             , np.full((RES_MAX_NATOMS-12-len( res.atoms.select_atoms("not (name P or name OP1 or name OP2 or name O5' or name C5' or name C4' or name O4' or name C3' or name O3' or name C2' or name O2' or name C1')") ), 3), np.nan)],axis=0) for res in u.residues] ))
    ###
	g = Data(x=th.tensor(res_feats, dtype=th.float), 
			edge_index=th.tensor([src_list, dst_list], dtype=th.int64), 
			pos=res_coods,
			edge_attr=th.tensor(np.array(edge_feats), dtype=th.float))#;print(res_coods.size())
	
	#g.ndata.pop("ca_pos")
	#g.ndata.pop("center_pos")
	#g.ndata["pos"] = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) for res in u.residues]))
	return g


def obtain_atom_pos(res, atom_name):
	cood = res.atoms.select_atoms(atom_name).positions
    
	if cood.size!=3:
		cood = np.full((1, 3), np.nan)
	    
	return cood


def obtain_ca_pos(res):
	if obtain_resname(res) == "M":
		return res.atoms.positions[0]
	else:
		try:
			pos = res.atoms.select_atoms("name CA").positions[0]
			return pos
		except:  ##some residues loss the CA atoms
			return res.atoms.positions.mean(axis=0)
        
        
def obtain_ca_pos_nuc(res):
	if obtain_resname(res) == "M":
		return res.atoms.positions[0]
	else:
		try:
			pos = res.atoms.select_atoms("name C5'").positions[0]
			return pos
		except:  ##some residues loss the CA atoms
			return res.atoms.positions.mean(axis=0)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
	try:
		#xx = res.atoms.select_atoms("not name H*")
		xx = res.atoms
		dists = distances.self_distance_array(xx.positions)
		ca = xx.select_atoms("name CA")
		c = xx.select_atoms("name C")
		n = xx.select_atoms("name N")
		o = xx.select_atoms("name O")
		return [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
	except:
		return [0, 0, 0, 0, 0]

    
def obtain_self_dist_nuc(res):
	try:
		#xx = res.atoms.select_atoms("not name H*")
		xx = res.atoms
		dists = distances.self_distance_array(xx.positions)#;print('atoms',len(xx),xx)
		o3_ = xx.select_atoms("name O3'")
		c3_ = xx.select_atoms("name C3'")
		c4_ = xx.select_atoms("name C4'")
		c5_ = xx.select_atoms("name C5'")
		o5_ = xx.select_atoms("name O5'")
		p = xx.select_atoms("name P")   
		c2_ = xx.select_atoms("name C2'")
		c1_ = xx.select_atoms("name C1'")
		o4_ = xx.select_atoms("name O4'")
        
        
		#nucatoms=[o3_, c3_, c4_, c5_, o5_, p, c2_, c1_, o4_]
		#dis=[ str( distances.dist(nucatoms[q],nucatoms[w])[-1][0]*0.1 ) for q in range(9) for w in range(q+1,9) ]
        
		#dis=','.join(dis)
		#print( str(dists.max()*0.1)+','+str(dists.min()*0.1)+','+dis )
        
		return [dists.max()*0.1, distances.dist(o3_,c5_)[-1][0]*0.1, distances.dist(o3_,o5_)[-1][0]*0.1, \
          distances.dist(o3_,p)[-1][0]*0.1, distances.dist(c3_,o5_)[-1][0]*0.1, distances.dist(c3_,p)[-1][0]*0.1, \
              distances.dist(c4_,p)[-1][0]*0.1, distances.dist(o3_,c1_)[-1][0]*0.1, distances.dist(o3_,o4_)[-1][0]*0.1, \
                  distances.dist(c5_,c1_)[-1][0]*0.1, distances.dist(c5_,c2_)[-1][0]*0.1]
	except:
		return [0]*11


def obtain_dihediral_angles(res):
	'''
	xx = res.atoms;from MDAnalysis.analysis import contacts
	ca = xx.select_atoms("name CA")
	c = xx.select_atoms("name C")
	n = xx.select_atoms("name N")
	o = xx.select_atoms("name O")
	cb = xx.select_atoms("name CB")
	cg1 = xx.select_atoms("name CG1")
    
	ch=n+ca+cb+cg1; ch2=n+cb+cg1+ca
	'''
	try:
		if res.phi_selection() is not None:
			phi = res.phi_selection().dihedral.value()
		else:
			phi = 0
		if res.psi_selection() is not None:
			psi = res.psi_selection().dihedral.value()
		else:
			psi = 0
		if res.omega_selection() is not None:
			omega = res.omega_selection().dihedral.value()
		else:
			omega = 0
		if res.chi1_selection() is not None:
			chi1 = res.chi1_selection().dihedral.value()#;print(res.chi1_selection());print(chi1);print(ch.dihedral.value());print(ch2.dihedral.value())
		else:
			chi1 = 0
		return [phi*0.01, psi*0.01, omega*0.01, chi1*0.01]
	except:
		return [0, 0, 0, 0]
    
    
def obtain_dihediral_angles_nuc(res_index, residues):
	res = residues[residues.resids == res_index]
	resforward = residues[residues.resids == res_index-1]
	resbackward = residues[residues.resids == res_index+1]
	#print( res, residues[residues.resids == res_index+1],  residues[residues.resids == res_index+1] is None )
    
	xx = res.atoms
	#main chain
	o3_ = xx.select_atoms("name O3'")
	c3_ = xx.select_atoms("name C3'")
	c4_ = xx.select_atoms("name C4'")
	c5_ = xx.select_atoms("name C5'")
	o5_ = xx.select_atoms("name O5'")
	p = xx.select_atoms("name P")   
	c2_ = xx.select_atoms("name C2'")
	c1_ = xx.select_atoms("name C1'")
	o4_ = xx.select_atoms("name O4'")
        
	o3_forward = resforward.atoms.select_atoms("name O3'")
    
	pbackward = resbackward.atoms.select_atoms("name P")
	o5_backward = resbackward.atoms.select_atoms("name O5'")
	#mainchain#
	d1 = o5_backward + pbackward + o3_ + c3_
	d2 = pbackward + o3_ + c3_ + c4_
	d3 = o3_ + c3_ + c4_ + c5_
	d4 = c3_ + c4_ + c5_ + o5_
	d5 = c4_ + c5_ + o5_ + p    
	d6 = c5_ + o5_ + p + o3_forward
	d7 = pbackward + o3_ + c3_ + c2_
	#sidechain
	d8 = o3_ + c3_ + c2_ + c1_
	d9 = c3_ + c2_ + c1_ + o4_
	d10 = c2_ + c1_ + o4_ + c4_
	d11 = c1_ + o4_ + c4_ + c5_
	d12 = o4_ + c4_ + c5_ + o5_
	d13 = o3_ + c3_ + c4_ + o4_
	d14 = c5_ + c4_ + c3_ + c2_
	d15 = c3_ + c4_ + o4_ + c1_
	d16 = c4_ + c3_ + c2_ + c1_

	#side chain  
	resname = obtain_resname_nuc(res)
	if resname == 'DA' or resname == 'A':
		n9 = xx.select_atoms("name N9")
		c4 = xx.select_atoms("name C4")
		d17 = c4_ + o4_ + c1_ + n9
		d18 = c3_ + c2_ + c1_ + n9 
		d19 = o4_ + c1_ + n9 + c4 
		d20 = c2_ + c1_ + n9 + c4 
	elif resname == 'DG' or resname == 'G':
		n9 = xx.select_atoms("name N9")
		c4 = xx.select_atoms("name C4")
		d17 = c4_ + o4_ + c1_ + n9
		d18 = c3_ + c2_ + c1_ + n9 
		d19 = o4_ + c1_ + n9 + c4 
		d20 = c2_ + c1_ + n9 + c4 
	elif resname == 'DI' or resname == 'I':
		n9 = xx.select_atoms("name N9")
		c4 = xx.select_atoms("name C4")
		d17 = c4_ + o4_ + c1_ + n9
		d18 = c3_ + c2_ + c1_ + n9 
		d19 = o4_ + c1_ + n9 + c4 
		d20 = c2_ + c1_ + n9 + c4
	elif resname == 'DC' or resname == 'C':
		n1 = xx.select_atoms("name N1")
		c2 = xx.select_atoms("name C2")
		d17 = c4_ + o4_ + c1_ + n1
		d18 = c3_ + c2_ + c1_ + n1
		d19 = o4_ + c1_ + n1 + c2 
		d20 = c2_ + c1_ + n1 + c2
	elif resname == 'DT' or resname == 'U' or resname == 'DU':
		n1 = xx.select_atoms("name N1")
		c2 = xx.select_atoms("name C2")
		d17 = c4_ + o4_ + c1_ + n1
		d18 = c3_ + c2_ + c1_ + n1
		d19 = o4_ + c1_ + n1 + c2 
		d20 = c2_ + c1_ + n1 + c2 
	else:
		d17 = c1_
		d18 = c1_
		d19 = c1_
		d20 = c1_ 
	'''
    ###non-standard classified as N
	elif resname == 'N':
		d17 = c1_
		d18 = c1_
		d19 = c1_
		d20 = c1_ 
        
	else:
		print('none', resname) #DI?
		return [0]*20
    ###non-standard classified as N
	'''        
	dname = [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20]
    #remove base atoms / can be realized in model 这样万一要用上也不用重新跑
	#dname = [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16]
	dvalue = []
	for i in dname:
		if len(i)==4:
			dvalue.append( i.dihedral.value()*0.01 )
		else:
			dvalue.append( 0 )
	
	return dvalue


def calc_res_features(res):
	return np.array(one_of_k_encoding_unk(obtain_resname(res), 
										['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +          #32  residue type	
			obtain_self_dist(res) +  #5
			obtain_dihediral_angles(res) #4		
			)


def calc_res_features_nuc(res_index, residues):
	res = residues[ residues.resids==res_index ]
	return np.array(one_of_k_encoding_unk(obtain_resname_nuc(res), 
										['A', 'G', 'C', 'U', 'I', 'N', 'DA', 'DG', 'DC', 'DT', 'DU', 'DI' ]) +          #12  residue type	
			obtain_self_dist_nuc(res) +  #11
			obtain_dihediral_angles_nuc(res_index, residues) #20		
			)


def obtain_resname(res):
	if res.resname[:2] == "CA":
		resname = "CA"
	elif res.resname[:2] == "FE":
		resname = "FE"
	elif res.resname[:2] == "CU":
		resname = "CU"
	else:
		resname = res.resname.strip()
	
	if resname in METAL:
		return "M"
	else:
		return resname
    
    
def obtain_resname_nuc(res):
	if res.resnames[0][:2] == "CA":
		resnames = "CA"
	elif res.resnames[0][:2] == "FE":
		resnames = "FE"
	elif res.resnames[0][:2] == "CU":
		resnames = "CU"
	else:
		resnames = res.resnames[0].strip()
        
    ###     non-standard classified as N
	if resnames not in ['A', 'G', 'C', 'U', 'I', 'DA', 'DG', 'DC', 'DT', 'DU', 'DI' ]:
		resnames = "N"
	###
	if resnames in METAL:
		return "M"
	else:
		return resnames
    

##'FE', 'SR', 'GA', 'IN', 'ZN', 'CU', 'MN', 'SR', 'K' ,'NI', 'NA', 'CD' 'MG','CO','HG', 'CS', 'CA',

def obatin_edge(u, cutoff=10.0):
	edgeids = []
	dismin = []
	dismax = []
	for res1, res2 in permutations(u.residues, 2):
		#rint(res1,res2)
		dist = calc_dist(res1, res2)
		if dist.min() <= cutoff:
			edgeids.append([res1.ix, res2.ix])
			dismin.append(dist.min()*0.1)
			dismax.append(dist.max()*0.1)
	return edgeids, np.array([dismin, dismax]).T


def obatin_edge_nuc(u, cutoff=10.0):
	edgeids = []
	dismin = []
	dismax = []
	for res1, res2 in permutations(u.residues, 2):
		dist = calc_dist_nuc(res1, res2)
		if dist.min() <= cutoff:
			edgeids.append([res1.ix, res2.ix])
			dismin.append(dist.min()*0.1)
			dismax.append(dist.max()*0.1)
	return edgeids, np.array([dismin, dismax]).T


def check_connect(u, i, j):
	if abs(i-j) != 1:
		return 0
	else:
		if i > j:
			i = j
		#print(i,j)

		nb1 = len(u.residues[i].get_connections("bonds"))
		nb2 = len(u.residues[i+1].get_connections("bonds"))
		nb3 = len(u.residues[i:i+2].get_connections("bonds"))
		if nb1 + nb2 == nb3 + 1:
			return 1
		else:
			return 0
        

def check_connect_nuc(u, i, j):
	if abs(i-j) != 1:
		return 0
	else:
		if i > j:
			i = j
 
		ures = [ele for ele in ob.OBResidueIter(u)]#;print(i)

		#res1_atoms = u.residues[i].select_atoms(res1)
		#res2_atoms = u.residues[i+1].select_atoms(res2)

		# Get the bonds in the system
		res1 = ures[i]
		res2 = ures[i+1]
		for atom1 in ob.OBResidueAtomIter(res1):
			for bond in ob.OBAtomBondIter(atom1):
				if bond.GetNbrAtom(atom1) in ob.OBResidueAtomIter(res2):
					return 1
				else:
					pass
        
		return 0

		#connected = ob.AreResiduesConnected(res1, res2);print(connected)
		#for bond in bonds:
		#	if (bond[0] in res1_atoms.indices and bond[1] in res2_atoms.indices) or (bond[1] in res1_atoms.indices and bond[0] in res2_atoms.indices):
		#	    return 1
        
		nb1 = len(u.residues[i].get_connections("bonds"))
		nb2 = len(u.residues[i+1].get_connections("bonds"))
		nb3 = len(u.residues[i:i+2].get_connections("bonds"))
		if nb1 + nb2 == nb3 + 1:
			return 1
		else:
			return 0
		
	

def calc_dist(res1, res2):
	#xx1 = res1.atoms.select_atoms('not name H*')
	#xx2 = res2.atoms.select_atoms('not name H*')
	#dist_array = distances.distance_array(xx1.positions,xx2.positions)
	dist_array = distances.distance_array(res1.atoms.positions,res2.atoms.positions)
    
	xx1 = res1.atoms.select_atoms("name N or name CA or name C or name O or name CB")
	xx2 = res2.atoms.select_atoms("name N or name CA or name C or name O or name CB")#;print(xx1.positions,xx2.positions)

	dist_array = distances.distance_array(xx1.positions,xx2.positions)
    
	return dist_array
	#return dist_array.max()*0.1, dist_array.min()*0.1


def calc_dist_nuc(res1, res2):
    #remove base atoms / keep backbone atoms
    if res1.resname.strip() in ['DA','A','DG','G','DI','I']:
        xx1 = res1.atoms.select_atoms("name P or name OP1 or name OP2 or name O5' or name C5' or name C4' or name O4' or name C3' or name O3' or name C2' or name O2' or name C1' or name N9")
    else:
        xx1 = res1.atoms.select_atoms("name P or name OP1 or name OP2 or name O5' or name C5' or name C4' or name O4' or name C3' or name O3' or name C2' or name O2' or name C1' or name N1")     
    	
    if res2.resname.strip() in ['DA','A','DG','G','DI','I']:
        xx2 = res2.atoms.select_atoms("name P or name OP1 or name OP2 or name O5' or name C5' or name C4' or name O4' or name C3' or name O3' or name C2' or name O2' or name C1' or name N9")
    else:
        xx2 = res2.atoms.select_atoms("name P or name OP1 or name OP2 or name O5' or name C5' or name C4' or name O4' or name C3' or name O3' or name C2' or name O2' or name C1' or name N1")     
    
    dist_array = distances.distance_array(xx1.positions,xx2.positions)
    return dist_array


def calc_atom_features(atom, explicit_H=False):
    """
    atom: rdkit.Chem.rdchem.Atom
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
       'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 
		'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 
		'Cu', 'Mn', 'Mo', 'other'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,'other']) + [atom.GetIsAromatic()]
                # [atom.GetIsAromatic()] # set all aromaticity feature blank.
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])	
    return np.array(results)


def calc_bond_features(bond, use_chirality=True):
    """
    bond: rdkit.Chem.rdchem.Bond
    use_chirality: whether to use chirality
    """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


	
def load_mol(molpath, explicit_H=False, use_chirality=True):
	# load mol
	if re.search(r'.pdb$', molpath):
		mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)
	elif re.search(r'.mol2$', molpath):
		mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)
	elif re.search(r'.sdf$', molpath):			
		mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H)
	else:
		raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")	
	
	if use_chirality:
		Chem.AssignStereochemistryFrom3D(mol)
	return mol


def mol_to_graph(mol, explicit_H=False, use_chirality=True):
	"""
	mol: rdkit.Chem.rdchem.Mol
	explicit_H: whether to use explicit H
	use_chirality: whether to use chirality
	"""   	
	# Add nodes
	num_atoms = mol.GetNumAtoms()
	
	atom_feats = np.array([calc_atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
	if use_chirality:
		chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
		chiral_arr = np.zeros([num_atoms,3]) 
		for (i, rs) in chiralcenters:
			if rs == 'R':
				chiral_arr[i, 0] =1 
			elif rs == 'S':
				chiral_arr[i, 1] =1 
			else:
				chiral_arr[i, 2] =1 
		atom_feats = np.concatenate([atom_feats,chiral_arr],axis=1)
	
	# obtain the positions of the atoms
	atomCoords = mol.GetConformer().GetPositions()
	
	# Add edges
	src_list = []
	dst_list = []
	bond_feats_all = []
	num_bonds = mol.GetNumBonds()
	for i in range(num_bonds):
		bond = mol.GetBondWithIdx(i)
		u = bond.GetBeginAtomIdx()
		v = bond.GetEndAtomIdx()
		bond_feats = calc_bond_features(bond, use_chirality=use_chirality)
		#bond_feats = np.concatenate([[1],bond_feats])
		src_list.extend([u, v])
		dst_list.extend([v, u])
		bond_feats_all.append(bond_feats)
		bond_feats_all.append(bond_feats)
		
	g = Data(x=th.tensor(atom_feats, dtype=th.float), 
			edge_index=th.tensor([src_list, dst_list]), 
			pos=th.tensor(atomCoords, dtype=th.float),
			edge_attr=th.tensor(np.array(bond_feats_all), dtype=th.float))
	
	return g



def mol_to_graph2(prot_path, lig_path, cutoff=10.0, explicit_H=False, use_chirality=True):
	prot = load_mol(prot_path, explicit_H=explicit_H, use_chirality=use_chirality) 
	lig = load_mol(lig_path, explicit_H=explicit_H, use_chirality=use_chirality)
	gp = prot_to_graph(prot, cutoff)
	gl = mol_to_graph(lig, explicit_H=explicit_H, use_chirality=use_chirality)
	return gp, gl


def label_query(pdbid, df):
	return df.loc[pdbid, "labels"]

def pdbbind_handle(pdbid, args):
	prot_path = "%s/%s/%s_prot/%s_p_pocket_%s.pdb"%(args.dir, pdbid, pdbid, pdbid, args.cutoff)
	lig_path = "%s/%s/%s_prot/%s_l.sdf"%(args.dir, pdbid, pdbid, pdbid)
	try:
		gp, gl = mol_to_graph2(prot_path, 
								lig_path, 
								cutoff=args.cutoff,
								explicit_H=args.useH, 
								use_chirality=args.use_chirality)
	except:
		print("%s failed to generare the graph"%pdbid)
		gp, gl = None, None
		#gm = None
	return pdbid, gp, gl, label_query(pdbid, args.ref)


def UserInput():
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('-d', '--dir', default=".",
						help='The directory to store the protein-ligand complexes.')	
	p.add_argument('-c', '--cutoff', default=None, type=float,
						help='the cutoff to determine the pocket')	
	p.add_argument('-o', '--outprefix', default="out",
						help='The output bin file.')	
	p.add_argument('-r', '--ref', default="/home/shenchao/pdbbind/pdbbind_2020_general.csv",
						help='The reference file to query the label of the complex.')	
	p.add_argument('-usH', '--useH', default=False, action="store_true",
						help='whether to use the explicit H atoms.')
	p.add_argument('-uschi', '--use_chirality', default=False, action="store_true",
						help='whether to use chirality.')							
	p.add_argument('-p', '--parallel', default=False, action="store_true",
						help='whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).')	
	
	args = p.parse_args()	
	return args



def main():
	args = UserInput()
	pdbids = [x for x in os.listdir(args.dir) if os.path.isdir("%s/%s"%(args.dir, x))]
	args.ref = pd.read_csv(args.ref, index_col=0, header=0)
	if args.parallel:
		results = Parallel(n_jobs=-1)(delayed(pdbbind_handle)(pdbid, args) for pdbid in pdbids)
	else:
		results = []
		for pdbid in pdbids:
			results.append(pdbbind_handle(pdbid, args))
	results = list(filter(lambda x: x[1] != None, results))
	ids, graphs_p, graphs_l, labels =  list(zip(*results))
	np.save("%s_ids"%args.outprefix, (ids, labels))
	th.save(graphs_p, "%s_prot.pt"%args.outprefix)
	th.save(graphs_l, "%s_lig.pt"%args.outprefix)
	


if __name__ == '__main__':
	main()

