import dgl
import torch
import scipy.sparse as sp
import numpy as np
from utils import eids_split, remove_all_edges_of_etype, get_num_nodes_dict

def process_data(g, 
                 val_ratio, 
                 test_ratio,
                 src_ntype = 'author', 
                 tgt_ntype = 'paper',
                 pred_etype = 'likes',
                 neg='pred_etype_neg'):
    '''
    Parameters
    ----------
    g : dgl graph
    
    val_ratio : float
    
    test_ratio : float
    
    src_ntype: string
        source node type
    
    tgt_ntype: string
        target node type

    neg: string
        One of ['pred_etype_neg', 'src_tgt_neg'], different negative sampling modes. See below.
    
    Returns
    ----------
    mp_g: 
        graph for message passing.
    
    graphs containing positive edges and negative edges for train, valid, and test
    '''
    
    u, v = g.edges(etype=pred_etype)
    src_N = g.num_nodes(src_ntype)
    tgt_N = g.num_nodes(tgt_ntype)

    M = u.shape[0] # number of directed edges
    eids = torch.arange(M)
    train_pos_eids, val_pos_eids, test_pos_eids = eids_split(eids, val_ratio, test_ratio)

    train_pos_u, train_pos_v = u[train_pos_eids], v[train_pos_eids]
    val_pos_u, val_pos_v = u[val_pos_eids], v[val_pos_eids]
    test_pos_u, test_pos_v = u[test_pos_eids], v[test_pos_eids]

    if neg == 'pred_etype_neg':
        # Edges not in pred_etype as negative edges
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(src_N, tgt_N))
        adj_neg = 1 - adj.todense()
        neg_u, neg_v = np.where(adj_neg != 0)
    elif neg == 'src_tgt_neg':
        # Edges not connecting src and tgt as negative edges
        
        # Collect all edges between the src and tgt
        src_tgt_indices = []
        for etype in g.canonical_etypes:
            if etype[0] == src_ntype and etype[2] == tgt_ntype:
                adj = g.adj(etype=etype)
                src_tgt_index = adj.coalesce().indices()        
                src_tgt_indices += [src_tgt_index]
        src_tgt_u, src_tgt_v = torch.cat(src_tgt_indices, dim=1)

        # Find all negative edges that are not in src_tgt_indices
        adj = sp.coo_matrix((np.ones(len(src_tgt_u)), (src_tgt_u.numpy(), src_tgt_v.numpy())), shape=(src_N, tgt_N))
        adj_neg = 1 - adj.todense()
        neg_u, neg_v = np.where(adj_neg != 0)
    else:
        raise ValueError('Unknow negative argument')
        
    neg_eids = np.random.choice(neg_u.shape[0], min(neg_u.shape[0], M), replace=False)
    train_neg_eids, val_neg_eids, test_neg_eids = eids_split(torch.from_numpy(neg_eids), val_ratio, test_ratio)

    # train_neg_u, train_neg_v = neg_u[train_neg_eids], neg_v[train_neg_eids]
    # val_neg_u, val_neg_v = neg_u[val_neg_eids], neg_v[val_neg_eids]
    # test_neg_u, test_neg_v = neg_u[test_neg_eids], neg_v[test_neg_eids]

    # Avoid losing dimension in single number slicing
    train_neg_u, train_neg_v = np.take(neg_u, train_neg_eids), np.take(neg_v, train_neg_eids)
    val_neg_u, val_neg_v = np.take(neg_u, val_neg_eids),np.take(neg_v, val_neg_eids)
    test_neg_u, test_neg_v = np.take(neg_u, test_neg_eids), np.take(neg_v, test_neg_eids)
    
    # Construct graphs
    pred_can_etype = (src_ntype, pred_etype, tgt_ntype)
    num_nodes_dict = get_num_nodes_dict(g)
    
    train_pos_g = dgl.heterograph({pred_can_etype: (train_pos_u, train_pos_v)}, num_nodes_dict)
    train_neg_g = dgl.heterograph({pred_can_etype: (train_neg_u, train_neg_v)}, num_nodes_dict)
    val_pos_g = dgl.heterograph({pred_can_etype: (val_pos_u, val_pos_v)}, num_nodes_dict)
    val_neg_g = dgl.heterograph({pred_can_etype: (val_neg_u, val_neg_v)}, num_nodes_dict)
    test_pos_g = dgl.heterograph({pred_can_etype: (test_pos_u, test_pos_v)}, num_nodes_dict)

    test_neg_g = dgl.heterograph({pred_can_etype: (test_neg_u, test_neg_v)}, num_nodes_dict)
    
    mp_g = remove_all_edges_of_etype(g, pred_etype) # Remove pred_etype edges but keep nodes
    return mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g


def load_dataset(dataset_dir, dataset_name, val_ratio, test_ratio):
    '''
    Parameters
    ----------
    dataset_dir : string
        dataset directory
    
    dataset_name : string
    
    val_ratio : float
    
    test_ratio : float

    Returns:
    ----------
    g: dgl graph
        The original graph

    processed_g: tuple of seven dgl graphs
        The outputs of the function `process_data`, 
        which includes g for message passing, train, valid, and test
        
    pred_pair_to_edge_labels : dict
        key=((source node type, source node id), (target node type, target node id))
        value=dict, {cannonical edge type: (source node ids, target node ids)}
        
    pred_pair_to_path_labels : dict 
        key=((source node type, source node id), (target node type, target node id))
        value=list of lists, each list contains (cannonical edge type, source node ids, target node ids)
    '''
    graph_saving_path = f'{dataset_dir}/{dataset_name}'
    graph_list, _ = dgl.load_graphs(graph_saving_path)
    pred_pair_to_edge_labels = torch.load(f'{graph_saving_path}_pred_pair_to_edge_labels')
    pred_pair_to_path_labels = torch.load(f'{graph_saving_path}_pred_pair_to_path_labels')
    g = graph_list[0]
    if 'synthetic' in dataset_name:
        src_ntype, tgt_ntype = 'user', 'item'
    elif 'citation' in dataset_name:
        src_ntype, tgt_ntype = 'author', 'paper'

    pred_etype = 'likes'
    neg = 'src_tgt_neg'
    processed_g = process_data(g, val_ratio, test_ratio, src_ntype, tgt_ntype, pred_etype, neg)
    return g, processed_g, pred_pair_to_edge_labels, pred_pair_to_path_labels

