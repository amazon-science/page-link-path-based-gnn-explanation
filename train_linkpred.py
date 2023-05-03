import os
import torch
import torch.nn.functional as F
import copy
import argparse
from sklearn.metrics import roc_auc_score
from pathlib import Path
from utils import set_seed, negative_sampling, print_args, set_config_args
from data_processing import load_dataset
from model import HeteroRGCN, HeteroLinkPredictionModel

parser = argparse.ArgumentParser(description='Train a GNN-based link prediction model')
parser.add_argument('--device_id', type=int, default=-1)

'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')
parser.add_argument('--dataset_name', type=str, default='aug_citation')
parser.add_argument('--valid_ratio', type=float, default=0.1) 
parser.add_argument('--test_ratio', type=float, default=0.2)

'''
GNN args
'''
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--out_dim', type=int, default=128)

'''
Link predictor args
'''
parser.add_argument('--src_ntype', type=str, default='user', help='prediction source node type')
parser.add_argument('--tgt_ntype', type=str, default='item', help='prediction target node type')
parser.add_argument('--pred_etype', type=str, default='likes', help='prediction edge type')
parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                   help='operation passed to dgl.EdgePredictor')
parser.add_argument('--lr', type=float, default=0.01, help='link predictor learning_rate') 
parser.add_argument('--num_epochs', type=int, default=200, help='How many epochs to train')
parser.add_argument('--eval_interval', type=int, default=1, help="Evaluate once per how many epochs")
parser.add_argument('--save_model', default=False, action='store_true', help='Whether to save the model')
parser.add_argument('--saved_model_dir', type=str, default='saved_models', help='Where to save the model')
parser.add_argument('--sample_neg_edges', default=False, action='store_true', 
                    help='If False, use fixed negative edges. If True, sample negative edges in each epoch')
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')

args = parser.parse_args()

if 'synthetic' in args.dataset_name:
    args.src_ntype = 'user'
    args.tgt_ntype = 'item'

elif 'citation' in args.dataset_name:
    args.src_ntype = 'author'
    args.tgt_ntype = 'paper'
    
if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')

if args.link_pred_op in ['cat']:
    pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
else:
    pred_kwargs = {}

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'train_eval')
    
print_args(args)

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    device = scores.device
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def run():
    set_seed(0)
    best_val_auc = 0
    pred_etype= args.pred_etype
    train_pos_src_nids, train_pos_tgt_nids = train_pos_g.edges(etype=pred_etype)            
    val_pos_src_nids, val_pos_tgt_nids = val_pos_g.edges(etype=pred_etype)            
    val_neg_src_nids, val_neg_tgt_nids = val_neg_g.edges(etype=pred_etype)            
    test_pos_src_nids, test_pos_tgt_nids = test_pos_g.edges(etype=pred_etype)            
    test_neg_src_nids, test_neg_tgt_nids = test_neg_g.edges(etype=pred_etype)            

    train_neg_src_nids, train_neg_tgt_nids = train_neg_g.edges(etype=pred_etype) 

    for epoch in range(1, args.num_epochs+1):
        train_pos_score = model(train_pos_src_nids, train_pos_tgt_nids, mp_g)   
        if args.sample_neg_edges:
            train_neg_src_nids, train_neg_tgt_nids = negative_sampling(train_pos_g, pred_etype) 
        train_neg_score = model(train_neg_src_nids, train_neg_tgt_nids, mp_g)
        loss = compute_loss(train_pos_score, train_neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                train_auc = compute_auc(train_pos_score, train_neg_score)
                val_pos_score = model(val_pos_src_nids, val_pos_tgt_nids, mp_g)
                val_neg_score = model(val_neg_src_nids, val_neg_tgt_nids, mp_g)
                val_auc = compute_auc(val_pos_score, val_neg_score)
                print('In epoch {}, loss: {:.4f}, train AUC: {:.4f}, val AUC: {:.4f}'.format(epoch, loss, train_auc, val_auc))
                if val_auc > best_val_auc:
                    best_epoch = epoch
                    best_val_auc = val_auc
                    state = copy.deepcopy(model.state_dict())

    with torch.no_grad():
        model.eval()
        model.load_state_dict(state)
        test_pos_score = model(test_pos_src_nids, test_pos_tgt_nids, mp_g)
        test_neg_score = model(test_neg_src_nids, test_neg_tgt_nids, mp_g)
        test_auc = compute_auc(test_pos_score, test_neg_score)
        print('Best epoch {}, val AUC: {:.4f}, test AUC: {:.4f}'.format(best_epoch, best_val_auc, test_auc))

processed_g = load_dataset(args.dataset_dir, args.dataset_name, args.valid_ratio, args.test_ratio)[1]
mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = [g.to(device) for g in processed_g]

encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
model = HeteroLinkPredictionModel(encoder, args.src_ntype, args.tgt_ntype, args.link_pred_op, **pred_kwargs)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

run()

if args.save_model:
    output_dir = Path.cwd().joinpath(args.saved_model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), output_dir.joinpath(f"{args.dataset_name}_model.pth"))




