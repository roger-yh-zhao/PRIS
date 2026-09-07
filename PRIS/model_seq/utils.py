import torch as th
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.distributions import Normal
from torch_scatter import scatter_add
from sklearn import metrics
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr


class EarlyStopping(object):
    """Early stop tracker
	
    Save model checkpoint when observing a performance improvement on
    the validation set and early stop if improvement has not been
    observed for a particular number of epochs.
	
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
	
    Examples
    --------
    Below gives a demo for a fake training process.
	
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.nn import MSELoss
    >>> from torch.optim import Adam
    >>> from dgllife.utils import EarlyStopping
	
    >>> model = nn.Linear(1, 1)
    >>> criterion = MSELoss()
    >>> # For MSE, the lower, the better
    >>> stopper = EarlyStopping(mode='lower', filename='test.pth')
    >>> optimizer = Adam(params=model.parameters(), lr=1e-3)
	
    >>> for epoch in range(1000):
    >>>     x = torch.randn(1, 1) # Fake input
    >>>     y = torch.randn(1, 1) # Fake label
    >>>     pred = model(x)
    >>>     loss = criterion(y, pred)
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     early_stop = stopper.step(loss.detach().data, model)
    >>>     if early_stop:
    >>>         break
	
    >>> # Load the final parameters saved by the model
    >>> stopper.load_checkpoint(model)
    """
    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            #dt = datetime.datetime.now()
            filename = 'early_stop.pth'
		
        if metric is not None:
            assert metric in ['rp', 'rs', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'rp' or 'rs' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['rp', 'rs', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'
		
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.timestep = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
	
    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
	
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
	
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score
	
    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
	
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
	
        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
	
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        self.timestep += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
	
    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
	
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        th.save({'model_state_dict': model.state_dict(),
                    'timestep': self.timestep}, self.filename)
	
    def load_checkpoint(self, model):
        '''Load the latest checkpoint
	
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(th.load(self.filename)['model_state_dict'])



def run_a_train_epoch(epoch, model, data_loader, optimizer, affi_weight=1.0, type_weight=1.0, aux_weight=0.001, dist_threhold=None, device='cpu'):
	model.train()
	total_loss = 0
	mdn_loss = 0
	affi_loss = 0
	nuc_type_loss = 0
	bond_loss = 0
	probs = []
	#nuc_types_split=[]
	#nuc_types_split4score=[]
	for batch_id, batch_data in enumerate(data_loader):
		#print(batch_data,flush=True)
		pdbids, bgp, bgl, labels = batch_data
		bgl, bgp = bgl.to(device), bgp.to(device)
		
		nuc_labels = th.argmax(bgl.x[:,:12], dim=1, keepdim=False)#;print('see',bgl.x[:,:12], nuc_labels)
		#bond_labels = th.argmax(bgp.x[:,:32], dim=1, keepdim=False)#;print('check',bgl.edge_attr[:,:4])
		#bond_labels = th.argmax(bgl.edge_attr[:,:4], dim=1, keepdim=False)
		bond_labels = bgl.edge_attr[:,:1].squeeze().to(th.int64)#;print(bond_labels);print(th.argmax(bgl.edge_attr[:,:4], dim=1, keepdim=False))#;quit()
		#bond_labels = th.argmax(bgl.edge_attr[:,:4], dim=1, keepdim=False)
		
		nuc_type_prointer, dist, nuc_types, bond_types, batch, R_batch, nuc_labels, nuc_labels2mdn = model(bgl, bgp)




		probs_nuc_types_mdn = nuc_type_prointer #mdn_discretizing(pi, sigma, mu, device)
		nuc_types_mdn = probs_nuc_types_mdn[th.where(dist <= dist_threhold)[0]]    
		nuc_labels_mdn = nuc_labels2mdn[th.where(dist <= dist_threhold)[0]]#;print( nuc_labels2mdn.size(), true.size() )
		nuc_labels_mdn = nuc_labels_mdn.squeeze(1)
		#nuc_mdn = F.cross_entropy(nuc_types_mdn, nuc_labels_mdn)
        
		R_batch_inter = R_batch[th.where(dist <= dist_threhold)[0]]
		weight_dist = dist[th.where(dist <= dist_threhold)[0]]
		nuc_types = inter_add_mdn(nuc_types, nuc_types_mdn, R_batch_inter, weight_dist)
        
		unique_inter_idx = th.unique(R_batch_inter)  
		all_idx = th.arange( nuc_types.size(0) )
		mask = th.ones( nuc_types.size(0), dtype=th.bool)
		mask[unique_inter_idx] = False
		noninter_idx = all_idx[mask]


		nuc = F.cross_entropy(nuc_types, nuc_labels)
		nuc_type = F.cross_entropy(nuc_types, nuc_labels)    
        
		if nuc_types.size(0)!=int(R_batch[-1])+1 or int(R_batch[0])!=0:
			print('wrong');quit()


		nuc_types4score = nuc_types.clone()   
		nuc_types4score[noninter_idx] = 0

        

		y = type2score( nuc_types, nuc_labels,  bgl.ptr) 

		batch = batch.to(device)#F.cross_entropy(prob_12, nuc_labels)
        

		labels = labels.float().type_as(y).to(device)
        
		if (affi_weight == 0.0):
			affi = 0
		else:
			affi = th.corrcoef(th.stack([y, labels]))[1,0]	
            
			#affi = F.mse_loss(y, labels);affi_weight = -affi_weight

            

		  
		bond = F.cross_entropy(bond_types, bond_labels)#;mdn = nuc
		loss = (affi * affi_weight) + (nuc_type * type_weight) + (bond * aux_weight)
		probs.append(y)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
        
		#affi_loss += affi.item() * batch.unique().size(0)/ len(data_loader.dataset);print(affi, affi_loss)
		#total_loss += loss.item() * batch.unique().size(0)
		#mdn_loss += mdn.item() * batch.unique().size(0)

		nuc_type_loss += nuc_type.item() * batch.unique().size(0)
		bond_loss += bond.item() * batch.unique().size(0)
		total_loss +=  nuc_type_loss * type_weight + bond_loss * aux_weight
		
		#print('Step, Total Loss: {:.3f}, MDN: {:.3f}'.format(total_loss, mdn_loss))
		if np.isinf(mdn_loss) or np.isnan(mdn_loss): break
		del bgl, bgp, nuc_labels, bond_labels, nuc_type_prointer, dist, nuc_types, bond_types, batch, nuc, bond, affi, loss, y
		th.cuda.empty_cache()
	
	ys = th.cat(probs)
	affi_loss = th.corrcoef(th.stack([ys, data_loader.dataset.labels.to(device)]))[1,0].item()
	
		
	return total_loss / len(data_loader.dataset)+ affi_loss* affi_weight, affi_loss, nuc_type_loss / len(data_loader.dataset), bond_loss / len(data_loader.dataset)



def run_an_eval_epoch(model, data_loader, pred=False, atom_contribution=False, res_contribution=False, dist_threhold=None,  affi_weight=1.0, type_weight=1.0, aux_weight=0.001, device='cpu'):
	model.eval()
	total_loss = 0
	affi_loss = 0
	nuc_type_loss = 0
	bond_loss = 0
	probs = []
	at_contrs = []
	res_contrs = []
	with th.no_grad():
		nuc_types_split=[]
		nuc_types4score_split=[]
		for batch_id, batch_data in enumerate(data_loader):
			pdbids, bgp, bgl, labels = batch_data
			bgl, bgp = bgl.to(device), bgp.to(device)
			nuc_labels = th.argmax(bgl.x[:,:12], dim=1, keepdim=False)
			#bond_labels = th.argmax(bgp.x[:,:32], dim=1, keepdim=False)
    		#bond_labels = th.argmax(bgl.edge_attr[:,:4], dim=1, keepdim=False)
			bond_labels = bgl.edge_attr[:,:1].squeeze().to(th.int64)
			
            
            
            
			nuc_type_prointer, dist, nuc_types, bond_types, batch, R_batch, nuc_labels, nuc_labels2mdn = model(bgl, bgp)
			probs_nuc_types_mdn = nuc_type_prointer 
			nuc_types_mdn = probs_nuc_types_mdn[th.where(dist <= dist_threhold)[0]]    
			nuc_labels_mdn = nuc_labels2mdn[th.where(dist <= dist_threhold)[0]]#;print( nuc_labels2mdn.size(), true.size() )
			nuc_labels_mdn = nuc_labels_mdn.squeeze(1)
			#nuc_mdn = F.cross_entropy(nuc_types_mdn, nuc_labels_mdn)

			R_batch_inter = R_batch[th.where(dist <= dist_threhold)[0]]
			weight_dist = dist[th.where(dist <= dist_threhold)[0]]
			nuc_types = inter_add_mdn(nuc_types, nuc_types_mdn, R_batch_inter, weight_dist)
                
			unique_inter_idx = th.unique(R_batch_inter)  
			all_idx = th.arange( nuc_types.size(0) )
			mask = th.ones( nuc_types.size(0), dtype=th.bool)
			mask[unique_inter_idx] = False
			noninter_idx = all_idx[mask]
            
			

             
			
			if pred or atom_contribution or res_contribution:
				#prob = calculate_probablity(nuc_type_prointer, nuc_labels2mdn)#;print(len(dist))
				#if dist_threhold is not None:
					#prob[th.where(dist > dist_threhold)[0]] = 0.
				
				batch = batch.to(device)
				if pred:
					#probx = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0))
                    
                    
					y = type2score( nuc_types, nuc_labels,  bgl.ptr)
					probs.append(y)#;print(prob.size(), probx) 
                    



					nuc_types4score = nuc_types.clone()
					nuc_types4score[noninter_idx] = 0
                    
					nuc_types2list=nuc_types.tolist()#;print(nuc_types2list);print(nuc_labels)
					nuc_types4score2list=nuc_types4score.tolist()#;print(nuc_types2list);print(nuc_labels)
					
					for ptri in range(len(bgl.ptr) - 1):
						#print( nuc_types[ int(bgl.ptr[ptri]):int(bgl.ptr[ptri+1]),:].size() )
						nuc_types_split.append(   nuc_types2list[ int(bgl.ptr[ptri]):int(bgl.ptr[ptri+1]) ]    )
						nuc_types4score_split.append(   nuc_types4score2list[ int(bgl.ptr[ptri]):int(bgl.ptr[ptri+1]) ]    )
					
                        

                    
				#if atom_contribution or res_contribution:				
					#contribs = [prob[batch==i].reshape(len(bgl.x[bgl.batch==i]), len(bgp.x[bgp.batch==i])) for i in th.arange(0, len(batch.unique()))]
					#if atom_contribution:
						#at_contrs.extend([contribs[i].sum(1).cpu().detach().numpy() for i in th.arange(0, len(batch.unique()))])
					#if res_contribution:
						#res_contrs.extend([contribs[i].sum(0).cpu().detach().numpy() for i in th.arange(0, len(batch.unique()))])
			
			else:
				mdn_weight=0
				if mdn_weight!=0:
				    mdn, prob = mdn_loss_fn(nuc_type_prointer, nuc_labels2mdn)
				    mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
				    mdn = mdn.mean()
				
				    batch = batch.to(device)	
				    if dist_threhold is not None:
					    prob = prob[th.where(dist <= dist_threhold)[0]] 
					    y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]] , dim=0, dim_size=batch.unique().size(0))
				    else:	
					    y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]] , dim=0, dim_size=batch.unique().size(0))
				#labels = labels.float().type_as(y).to(device)
				
				#affi = F.mse_loss(y, labels)	
              
				#nuc_type = F.cross_entropy(nuc_types_inter, nuc_labels_inter)
				#nuc = F.cross_entropy(nuc_types_noninter, nuc_labels_noninter)  
				y = type2score( nuc_types, nuc_labels,  bgl.ptr)
                
				nuc = F.cross_entropy(nuc_types, nuc_labels)
				nuc_type = F.cross_entropy(nuc_types, nuc_labels)  
                
				bond = F.cross_entropy(bond_types, bond_labels);mdn = nuc
				loss =  (nuc_type * type_weight) + (bond * aux_weight)
				#loss = mdn + affi * affi_weight + (nuc * aux_weight) + (bond * aux_weight)
				
				probs.append(y)
				
				total_loss += loss.item() * batch.unique().size(0)
				nuc_type_loss += nuc_type.item() * batch.unique().size(0)
				bond_loss += bond.item() * batch.unique().size(0)			
			
			del bgl, bgp, nuc_labels, bond_labels, nuc_type_prointer, nuc_types, bond_types, batch 
			th.cuda.empty_cache()
	
	if atom_contribution or res_contribution:
		if pred:
			preds = th.cat(probs)
			return [preds.cpu().detach().numpy(),at_contrs,res_contrs]
		else:
			return [None, at_contrs,res_contrs]
	else:
		if pred:
			preds = th.cat(probs)#;print(preds.cpu().detach().numpy() )#, nuc_types_split)
			return preds.cpu().detach().numpy(), nuc_types_split, nuc_types4score_split
		else:		
			ys = th.cat(probs)
			affi_loss = th.corrcoef(th.stack([ys, data_loader.dataset.labels.to(device)]))[1,0].item()
			#del ys
			return total_loss / len(data_loader.dataset)+ affi_loss * affi_weight, affi_loss, nuc_type_loss / len(data_loader.dataset),  bond_loss / len(data_loader.dataset)


def set_random_seed(seed=10):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    #th.backends.cudnn.benchmark = False
    #th.backends.cudnn.deterministic = True
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)



def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += th.log(pi)
    prob = logprob.exp().sum(1)
	
    return prob



def mdn_loss_fn(pi, sigma, mu, y, eps1=1e-10, eps2=1e-10):
    normal = Normal(mu, sigma)
    #loss = th.exp(normal.log_prob(y.expand_as(normal.loc)))
    #loss = th.sum(loss * pi, dim=1)
    #loss = -th.log(loss)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    #loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)
    prob = (th.log(pi + eps1) + loglik).exp().sum(1)
    loss = -th.log(prob + eps2)
    return loss, prob


def nomdn_loss_fn(nuc_type_prointer, y, eps1=1e-10, eps2=1e-10):


    #y = y.long().view(-1)


    if nuc_type_prointer.ndim != 2:
        raise ValueError(
            f"nuc_type_prointer should be [N, 12],"
            f"now shape is {nuc_type_prointer.shape}"
        )

    if nuc_type_prointer.size(1) != 12:
        raise ValueError(
            f"type shoule be 12, now is {nuc_type_prointer.size(1)}"
        )

    if nuc_type_prointer.size(0) != y.size(0):
        raise ValueError(
            "nuc_type_prointer and y is different:"
            f"{nuc_type_prointer.size(0)} vs {y.size(0)}"
        )


    positive_score = nuc_type_prointer + eps1


    # class_prob.shape = [N, 12]
    class_prob = positive_score / positive_score.sum(
        dim=1,
        keepdim=True
    )


    # y.unsqueeze(1).shape = [N, 1]
    # prob.shape = [N]
    prob = class_prob.gather(
        dim=1,
        index=y.unsqueeze(1)
    ).squeeze(1)


    loss = -th.log(prob + eps2)

    return loss, prob




def mdn_discretizing(pi, sigma, mu, device):

    probs = th.zeros( pi.size(0),12 , device=device )
    for i in range(12):
        y = th.full((pi.size(0), 1), fill_value=i, dtype=th.int64 , device=device)
        loss,prob = mdn_loss_fn(pi, sigma, mu,y)
        probs[:, i] = prob.squeeze(-1)

    #print(probs);print(prob);quit();
    return probs

def inter_add_mdn(nuc_types, nuc_types_mdn, R_batch_inter, weight):

    nuc_types = nuc_types*0.02
    for i, idx in enumerate(R_batch_inter):

        #print(nuc_types[idx])
        #nuc_types[idx] += nuc_types_mdn[i]*100/weight[i]#;print( nuc_types_mdn[i],  weight[i] )
        #nuc_types[idx] += nuc_types_mdn[i]*30#;print( nuc_types_mdn[i],  weight[i] )
        nuc_types[idx] += nuc_types_mdn[i]#;print( nuc_types_mdn[i],  weight[i] )
        #print(nuc_types[idx])    
    
    return nuc_types


def type2score( nuc_types4score, nuc_labels, batch ):
	rows = th.arange(nuc_types4score.size(0), device=nuc_types4score.device)
	selected = nuc_types4score[rows, nuc_labels].unsqueeze(1)#;print( nuc_types4score[10:30] , selected[10:30], nuc_labels[10:30] );quit()    
	#print(selected.size(), batch.size() );quit()
    
	segment_sum = th.stack(  [selected[batch[i]:batch[i+1]].sum(dim=0) for i in range(len(batch) - 1)]  )
                
	return segment_sum.squeeze(1)