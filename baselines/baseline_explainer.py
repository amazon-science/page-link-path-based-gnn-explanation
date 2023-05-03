import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from utils import hetero_src_tgt_khop_in_subgraph

def get_edge_mask_dict(ghetero):
    '''
    Create a dictionary mapping etypes to learnable edge masks 
            
    Parameters
    ----------
    ghetero : heterogeneous dgl graph.

    Return
    ----------
    edge_mask_dict : dictionary
        key=etype, value=torch.nn.Parameter with size number of etype edges
    '''
    device = ghetero.device
    edge_mask_dict = {}
    for etype in ghetero.canonical_etypes:
        num_edges = ghetero.num_edges(etype)
        num_nodes = ghetero.edge_type_subgraph([etype]).num_nodes()

        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * num_nodes))
        edge_mask_dict[etype] = torch.nn.Parameter(torch.randn(num_edges, device=device) * std)
    return edge_mask_dict


class HeteroGNNExplainer(nn.Module):
    """GNNExplainer for heterogeneous link prediction explanation

    Adapted from the DGL GNNExplainer implementation
    https://docs.dgl.ai/en/0.8.x/_modules/dgl/nn/pytorch/explain/gnnexplainer.html#GNNExplainer

    Parameters
    ----------
    model : nn.Module
        The GNN model to explain.

        * The required arguments of its forward function are graph and feat.
          The latter one is for input node features.
        * It should also optionally take an eweight argument for edge weights
          and multiply the messages by it in message passing.
        * The output of its forward function is the logits for the predicted
          node/graph classes.
    lr : float, optional
        The learning rate to use, default to 0.01.
    num_epochs : int, optional
        The number of epochs to train.
    alpha1 : float, optional
        A higher value will make the explanation edge masks more sparse by decreasing
        the sum of the edge mask.
    alpha2 : float, optional
        A higher value will make the explanation edge masks more discrete by decreasing
        the entropy of the edge mask.
    log : bool, optional
        If True, it will log the computation process, default to True.
    """
    def __init__(self,
                 model,
                 lr=0.001,
                 num_epochs=100,
                 *,
                 alpha1=0.005,
                 alpha2=1.0,
                 log=False):
        super(HeteroGNNExplainer, self).__init__()
        self.model = model
        self.src_ntype = model.src_ntype
        self.tgt_ntype = model.tgt_ntype
        self.lr = lr
        self.num_epochs = num_epochs
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.log = log

        self.all_loss = defaultdict(list)

    def _init_masks(self, ghetero):
        r"""Initialize learnable edge mask.
        
        Parameters
        ----------
        ghetero : heterogeneous dgl graph.

        Returns
        -------
        edge_mask_dict : dictionary
            key=etype, value=torch.nn.Parameter with size number of etype edges
         """
        edge_mask_dict = get_edge_mask_dict(ghetero)
        return edge_mask_dict

    def _loss_regularize(self, loss, eweights):
        r"""Add regularization terms to the loss.

        Parameters
        ----------
        loss : Tensor
            Loss value.
        
        eweights : Tensor
            Edge mask of shape :math:`(E)`, where :math:`E`
            is the number of edges.

        Returns
        -------
        Tensor
            Loss value with regularization terms added.
        """
        # epsilon for numerical stability
        eps = 1e-15

        # Edge mask sparsity regularization
        reg1 = torch.sum(eweights)
        # reg1 = torch.mean(eweights)
        loss = loss + self.alpha1 * reg1
        
        # Edge mask entropy regularization
        ent = - eweights * torch.log(eweights + eps) - \
            (1 - eweights) * torch.log(1 - eweights + eps)
        reg2 = ent.mean()
        loss = loss + self.alpha2 * reg2

        self.all_loss['reg1'] += [reg1.item()]
        self.all_loss['reg2'] += [reg2.item()]
        
        return loss
    
    def get_comp_g_edge_mask(self, src_nid, tgt_nid, ghetero, feat_nids):
        """
        Get the explanation mask for the computation graph.
        
        Parameters
        ----------
        src_nid : int
            The source node of the link.
        tgt_nid : int
            The target node of the link.
        ghetero : DGLGraph
            A heterogeneous graph.
        feat_nids : tensor
            Node ids of the node feature. Passed into the model.forward

        Return:
        -------
        edge_mask_dict : dictionary
            key=etype, value=torch.nn.Parameter with size number of etype edges
        """
            
        self.model.eval()
        # Get the initial prediction.
        with torch.no_grad():
            score = self.model(src_nid, tgt_nid, ghetero, feat_nids)
            pred = (score > 0).int().item()
        
        edge_mask_dict = self._init_masks(ghetero)
        optimizer = torch.optim.Adam(edge_mask_dict.values(), lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)

        eweight_norm = 0
        for e in range(self.num_epochs):
            # apply sigmoid to edge_mask to get eweight
            eweight_dict = {etype: edge_mask_dict[etype].sigmoid() for etype in edge_mask_dict}
            
            score = self.model(src_nid, tgt_nid, ghetero, feat_nids, eweight_dict=eweight_dict)

            loss = (-1) ** pred * score.sigmoid().log()

            self.all_loss['loss'] += [loss.item()]
        
            eweights = torch.cat(list(edge_mask_dict.values())).sigmoid()
            
            curr_eweight_norm = eweights.norm()
            eweight_norm = curr_eweight_norm

            loss = self._loss_regularize(loss, eweights)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.all_loss['total_loss'] += [loss.item()]

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        edge_mask_dict = {k : v.detach() for k, v in edge_mask_dict.items()}
        return edge_mask_dict
    
    def explain(self, src_nid, tgt_nid, ghetero, num_hops=2):
        # Extract the computation graph (k-hop subgraph)
        (comp_g_src_nid, 
         comp_g_tgt_nid, 
         comp_g, 
         comp_g_feat_nids) = hetero_src_tgt_khop_in_subgraph(self.src_ntype, 
                                                             src_nid, 
                                                             self.tgt_ntype, 
                                                             tgt_nid, 
                                                             ghetero, 
                                                             num_hops)
        # Get the explanation on the computation graph
        edge_mask_dict = self.get_comp_g_edge_mask(comp_g_src_nid, 
                                                   comp_g_tgt_nid, 
                                                   comp_g, 
                                                   comp_g_feat_nids) 

        return edge_mask_dict


class HeteroPGExplainer(nn.Module):
    """PGExplainer for heterogeneous link prediction explanation

    Adapted from the DIG PGExplainer implementation
    https://github.com/divelab/DIG/blob/dig/dig/xgraph/method/pgexplainer.py
    
    Parameters
    ----------
    model : nn.Module
        The GNN model to explain.

        * The required arguments of its forward function are graph and feat.
          The latter one is for input node features.
        * It should also optionally take an eweight argument for edge weights
          and multiply the messages by it in message passing.
        * The output of its forward function is the logits for the predicted
          node/graph classes.
    num_hops : int
        The number of hops for GNN information aggregation.

    in_dim : int
        The dimension of the node representation generated by the model.encoder.
        It will be used as the input dimension for the mask generator.

    mask_generator_hidden_dim: int
        The hidden dimension of the mask generator.
    
    lr : float, optional
        The learning rate to use, default to 0.01.
    num_epochs : int, optional
        The number of epochs to train.
    alpha1 : float, optional
        A higher value will make the explanation edge masks more sparse by decreasing
        the sum of the edge mask.
    alpha2 : float, optional
        A higher value will make the explanation edge masks more discrete by decreasing
        the entropy of the edge mask.
        
    t0 : The temperature at the first epoch when training the mask generator
    t1 : The temperature at the final epoch when training the mask generator
    sample_bias: bias when sampling from the soft mask.
    """

    def __init__(self,
                 model,
                 num_hops,
                 ghetero,
                 in_dim,
                 mask_generator_hidden_dim=64,
                 lr=0.005,
                 num_epochs=20,
                 *,
                 alpha1=0.01,
                 alpha2=5e-4,
                 t0=5.0,
                 t1=1.0,
                 sample_bias=0.0):
        super(HeteroPGExplainer, self).__init__()
        self.model = model
        self.src_ntype = model.src_ntype
        self.tgt_ntype = model.tgt_ntype
        self.num_hops = num_hops
        self.in_dim = in_dim
        self.mg_hidden_dim = mask_generator_hidden_dim
        self.lr = lr
        self.num_epochs = num_epochs
        
        self._init_mask_generator(ghetero)
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.t0 = t0
        self.t1 = t1
        self.sample_bias = sample_bias
    
        self.all_loss = defaultdict(list)
        self.epoch_loss = defaultdict(list)
    
    def _init_mask_generator(self, ghetero):
        """
        Initialize learnable edge mask generators
        
        Parameters
        ----------
        ghetero : heterogeneous dgl graph.

        """
        self.mask_generators = {}
        device = ghetero.device
        for etype in ghetero.canonical_etypes:
            mask_generator = nn.ModuleList()
            mask_generator.append(nn.Sequential(nn.Linear(self.in_dim*4, self.mg_hidden_dim), nn.ReLU()))
            mask_generator.append(nn.Linear(self.mg_hidden_dim, 1))
            self.mask_generators[etype] = mask_generator.to(device)
            
    def _loss_regularize(self, loss, eweights):
        r"""Add regularization terms to the loss.

        Parameters
        ----------
        loss : Tensor
            Loss value.
        eweights : Tensor
            Edge mask of shape :math:`(E)`, where :math:`E`
            is the number of edges.

        Returns
        -------
        Tensor
            Loss value with regularization terms added.
        """
        # epsilon for numerical stability
        eps = 1e-15

        # Edge mask sparsity regularization
        reg1 = torch.sum(eweights)
        loss = loss + self.alpha1 * reg1
        
        # Edge mask entropy regularization
        ent = - eweights * torch.log(eweights + eps) - \
            (1 - eweights) * torch.log(1 - eweights + eps)
        reg2 = ent.mean()
        loss = loss + self.alpha2 * reg2

        self.epoch_loss['reg1'] += [reg1.item()]
        self.epoch_loss['reg2'] += [reg2.item()]
        
        return loss
    
    def concrete_sample(self, log_alpha, beta = 1.0, training = True):
        """ 
        Sample from the instantiation of concrete distribution when training 
        
        see https://github.com/divelab/DIG/blob/dig/dig/xgraph/method/pgexplainer.py
        
        """
        if training:
            bias = self.sample_bias
            random_noise = torch.rand(log_alpha.shape) * (1 - 2 * bias) + bias
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            # gate_inputs = gate_inputs.sigmoid()
            gate_inputs = gate_inputs

        else:
            # gate_inputs = log_alpha.sigmoid()
            gate_inputs = log_alpha
        return gate_inputs

    def train_mask_generator(self, 
                             train_src_nids, 
                             train_tgt_nids, 
                             ghetero, 
                             tmp=1.0,
                             batch_size=0):
        """ 
        Train a mask generator for heterogeneous edge types
        
        Parameters
        ----------
        train_src_nids : tensor
            The source node of the links for training
        
        train_tgt_nids : tensor
            The target node of the links for training
       
        ghetero : heterogeneous dgl graph.
        
        tmp: float
            The temperature parameter fed to the sample procedure

        batch_size : int

        Returns
        -------
        None
        """
        params = sum([list(mg.parameters()) for mg in self.mask_generators.values()], [])
        optimizer = torch.optim.Adam(params, lr=self.lr)
        
        with torch.no_grad():
            self.model.eval()
            preds = (self.model(train_src_nids, train_tgt_nids, ghetero) > 0).int()
        
        for mg in self.mask_generators.values():
            mg.train()
            
        # Train the mask generator
        for epoch in tqdm(range(1, self.num_epochs+1)):
            self.epoch_loss['reg1'] = []
            self.epoch_loss['reg2'] = []
            self.epoch_loss['loss'] = []
            self.epoch_loss['total_loss'] = []
            
            total_loss = 0.0
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.num_epochs))
            optimizer.zero_grad()
            
            if batch_size > 0:
                # randomly sample small batches for training
                indices = np.random.choice(train_src_nids.shape[0], batch_size, replace=False)
            else:
                indices = range(train_src_nids.shape[0])
                
            for i in indices:
                src_nid, tgt_nid = train_src_nids[i], train_tgt_nids[i]
                
                # Extract the computation graph (k-hop subgraph)
                (comp_g_src_nid, 
                 comp_g_tgt_nid, 
                 comp_g, 
                 comp_g_feat_nids) = hetero_src_tgt_khop_in_subgraph(self.src_ntype, 
                                                                     src_nid, 
                                                                     self.tgt_ntype, 
                                                                     tgt_nid, 
                                                                     ghetero, 
                                                                     self.num_hops)

                # Get the explanation on the computation graph
                edge_mask_dict = self.get_comp_g_edge_mask(comp_g_src_nid, 
                                                           comp_g_tgt_nid, 
                                                           comp_g, 
                                                           comp_g_feat_nids, 
                                                           tmp, 
                                                           training=True)
                
                eweight_dict = {etype: edge_mask_dict[etype].sigmoid() for etype in edge_mask_dict}

                score = self.model(comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids, eweight_dict=eweight_dict)
                loss = (-1) ** preds[i] * score.sigmoid().log()

                self.epoch_loss['loss'] += [loss.item()]
                
                # Add regularizer
                eweights = torch.cat(list(edge_mask_dict.values())).sigmoid()
                loss = self._loss_regularize(loss, eweights)
            
                loss.backward()
                
                total_loss += loss.item()
                self.epoch_loss['total_loss'] += [loss.item()]

            optimizer.step()
                
            self.all_loss['reg1'] += [np.mean(self.epoch_loss['reg1'])]
            self.all_loss['reg2'] += [np.mean(self.epoch_loss['reg2'])]
            self.all_loss['loss'] += [np.mean(self.epoch_loss['loss'])]
            self.all_loss['total_loss'] += [np.mean(self.epoch_loss['total_loss'])]
            
            #print(f'Epoch: {epoch} | Loss: {total_loss/preds.shape[0]}')      

    def get_comp_g_edge_mask(self, src_nid, tgt_nid, ghetero, feat_nids, tmp = 1.0, training=False):
        """
        Get the explanation mask for the computation graph.
        
        Parameters
        ----------
        src_nid : int
            The source node of the link.
        tgt_nid : int
            The target node of the link.
        ghetero : a heterogeneous dgl graph
        
        feat_nids : tensor
            Node ids of the node feature. Passed into the model.forward
        tmp : float 
            The temperature parameter fed to the sample procedure
        training : bool
            Whether in training procedure or not
        Return:
        -------
        edge_mask_dict : dictionary
            key=etype, value=torch.nn.Parameter with size number of etype edges
        """
        self.model.eval()
        
        with torch.no_grad():
            h = self.model.encode(ghetero, feat_nids)
            
        edge_mask_dict = {}
        
        src_h = h[self.src_ntype][src_nid]
        tgt_h = h[self.tgt_ntype][tgt_nid]
        for can_etype in ghetero.canonical_etypes:
            u_ntype, etype, v_ntype = can_etype
            #num_u = ghetero.num_nodes(ntype=u_ntype)
            #num_v = ghetero.num_nodes(ntype=v_ntype)
            u, v = ghetero.edges(etype=etype)
            f1 = h[u_ntype][u]
            f2 = h[v_ntype][v]
            
            src_f = src_h.repeat(f1.shape[0], 1)
            tgt_f = tgt_h.repeat(f2.shape[0], 1)
            
            # concatenate the source h and target h at the end
            f_cat = torch.cat([f1, f2, src_f, tgt_f], dim=-1)  
            edge_h = f_cat
            for elayer in self.mask_generators[can_etype]:
                edge_h = elayer(edge_h)

            values = edge_h.reshape(-1)
            values = self.concrete_sample(values, beta=tmp, training=training)
            
            #sparse_edges = torch.cat([u.unsqueeze(0), v.unsqueeze(0)])
            #mask_sparse = torch.sparse_coo_tensor(
            #    sparse_edges, values, (num_u, num_v)
            #)
            #mask_dense = mask_sparse.to_dense()
            ## set the symmetric edge weights
            #edge_mask = mask_dense[sparse_edges[0], sparse_edges[1]]
            #edge_mask_dict[can_etype] = edge_mask
            
            edge_mask_dict[can_etype] = values
            
        return edge_mask_dict
    
    def explain(self, src_nid, tgt_nid, ghetero, tmp = 1.0):
        # Extract the computation graph (k-hop subgraph) 
        (comp_g_src_nid, 
         comp_g_tgt_nid, 
         comp_g, 
         comp_g_feat_nids) = hetero_src_tgt_khop_in_subgraph(self.src_ntype, 
                                                             src_nid, 
                                                             self.tgt_ntype, 
                                                             tgt_nid, 
                                                             ghetero, 
                                                             self.num_hops)
        # Get the explanation on the computation graph
        edge_mask_dict = self.get_comp_g_edge_mask(comp_g_src_nid, 
                                                   comp_g_tgt_nid, 
                                                   comp_g, 
                                                   comp_g_feat_nids, 
                                                   tmp)
        return edge_mask_dict







