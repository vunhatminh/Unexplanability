import numpy as np
import torch_geometric.utils.convert as convert

def get_node_gt(node_idx, motif_size, offset = 0):
    start_idx = int((node_idx-offset)/motif_size)*motif_size + offset
    last_idx = start_idx + motif_size
    return range(start_idx, last_idx)

def get_edge_gt(node_idx, edge_index, motif_size, offset):
    src = get_node_gt(node_idx, motif_size, offset)
    
    gt_edge = []
    for edge_id in range(edge_index.shape[1]):
        node_0 = edge_index[0][edge_id]
        node_1 = edge_index[1][edge_id]
        
        if node_0 in src:
            if node_1 in src:
                gt_edge.append(edge_id)
                
    gt = []
    for e in gt_edge:
        gt.append((edge_index[0,e].cpu().detach().item(),edge_index[1,e].cpu().detach().item()))
    
    return gt

def edge_to_node_score(edge_index, edge_score, num_nodes):
    node_score = {}
    node_count = {}
    for node in range(num_nodes):
        node_score[node] = 0
        node_count[node] = 0
    
    for edge_id in range(edge_index.shape[1]):
        edge_0 = edge_index[0][edge_id]
        edge_1 = edge_index[1][edge_id]

        if edge_score[edge_id] > node_score[int(edge_0)]:
            node_score[int(edge_0)] = edge_score[edge_id]
        if edge_score[edge_id] > node_score[int(edge_1)]:
            node_score[int(edge_1)] = edge_score[edge_id]

    return node_score

def node_to_edge_score(edge_index, node_score):
    
    edge_score = {}
    for edge in range(edge_index.shape[1]):
        edge_score[edge] = 0
        
    for edge_id in range(edge_index.shape[1]):
        node_0 = edge_index[0][edge_id]
        node_1 = edge_index[1][edge_id]
        
        if node_score[node_0] > edge_score[edge_id]:
            edge_score[edge_id] = node_score[node_0]
        if node_score[node_1] > edge_score[edge_id]:
            edge_score[edge_id] = node_score[node_1]
    
    score = []
    for edge in range(edge_index.shape[1]):
        score.append(edge_score[edge])
    
    return score

def get_node_gt_score_from_dict(node_idx, node_score_dict):
    score = []
    for node in get_node_gt(node_idx):
        score.append(node_score_dict[node])
    return np.asarray(score)

def TP_node_score(node_score, node_idx, motif_size, offset):
    return sum(node_score[get_node_gt(node_idx, motif_size, offset)])/len(get_node_gt(node_idx, motif_size, offset))

def FP_node_score(node_score, node_idx, motif_size, offset):
    others = list(set(range(len(node_score))).difference(get_node_gt(node_idx, motif_size, offset)))
    return sum(node_score[others])/len(get_node_gt(node_idx, motif_size, offset))

def TP_edge_score(edge_score, edge_index, node_idx, motif_size, offset):
    gt_edge = get_edge_gt(node_idx, edge_index, motif_size, offset)
    return sum(edge_score[gt_edge])/len(gt_edge)

def FP_edge_score(edge_score, edge_index, node_idx, motif_size, offset):
    gt_edge = get_edge_gt(node_idx, edge_index, motif_size, offset)
    others = list(set(range(edge_index.shape[1])).difference(gt_edge))
    return sum(edge_score[others])/len(gt_edge)

def TP_node_discrete(node_score, node_idx, motif_size, offset):
    gt = get_node_gt(node_idx, motif_size, offset)
    num_gt = len(gt)
    selected = np.argpartition(node_score, -num_gt)[-num_gt:]
    tp = set(gt).intersection(selected)
                  
    return len(tp)/num_gt
                  
def TP_edge_discrete(edge_score, edge_index, node_idx, motif_size, offset):
    gt_edge = get_edge_gt(node_idx, edge_index, motif_size, offset)
    num_gt = len(gt_edge)
    selected = np.argpartition(edge_score, -num_gt)[-num_gt:]
    tp = set(gt_edge).intersection(selected)          
    return len(tp)/num_gt

def get_node_gt_dict(node_range, motif_size, offset):
    node_gt_dict = {}
    for node in node_range:
        node_gt_dict[node] = get_node_gt(node, motif_size, offset)
    return node_gt_dict

def get_edge_gt_dict(node_range, edge_index, motif_size, offset):
    edge_gt_dict = {}
    for node in node_range:
        edge_gt_dict[node] = get_edge_gt(node, edge_index, motif_size, offset)
    return edge_gt_dict

def TP_node_discrete_from_dict(node_score, node_idx, node_gt_dict):
    gt = node_gt_dict[node_idx]
    num_gt = len(gt)
    selected = np.argpartition(node_score, -num_gt)[-num_gt:]
    tp = set(gt).intersection(selected)
                  
    return len(tp)/num_gt
                  
def TP_edge_discrete_from_dict(edge_score_mat, node_idx, edge_gt_dict):
    gt_edge = edge_gt_dict[node_idx]    
    num_gt = len(gt_edge)
    selected = np.c_[np.unravel_index(np.argpartition(edge_score_mat.toarray().ravel(),-num_gt)[-num_gt:],edge_score_mat.shape)]         
    tp = set(gt_edge).intersection([tuple(e) for e in selected.tolist()]) 
    return len(tp)/num_gt