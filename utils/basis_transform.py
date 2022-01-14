import torch
import dgl
from torch_geometric.utils import to_dense_adj
# from utils.dense import to_dense_adj


def basis_transform(g,
                    norm,
                    power,
                    epsilon,
                    identity):
    (e_idx0, e_idx1) = g.edges()
    edge_idx = torch.stack([e_idx0, e_idx1], dim=0)
    adj = to_dense_adj(edge_idx).squeeze(0)  # Graphs may have only one node.

    if norm == 'Sym' and identity == 0:
        adj = adj - torch.eye(adj.shape[0], dtype=adj.dtype) * (1 - identity)
        deg_nopad = adj.sum(1)
        isolated = torch.where(deg_nopad <= 1e-6, torch.ones_like(deg_nopad), torch.zeros_like(deg_nopad))
        if isolated.sum(0) != 0:
            adj = adj + isolated.diag_embed()

    if isinstance(epsilon, str):
        eps = epsilon.split('d')
        epsilon = float(eps[0]) / float(eps[1])

    if norm == 'Eps':
        eig_val, eig_vec = torch.linalg.eigh(adj)
        padding = torch.ones_like(eig_val)
        eig_sign = torch.where(eig_val >= 0, padding, padding * -1)
        eig_val_nosign = eig_val.abs()
        eig_val_nosign = torch.where(eig_val_nosign > 1e-6, eig_val_nosign, torch.zeros_like(eig_val_nosign))  # Precision limitation
        eig_val_smoothed = eig_val_nosign.pow(epsilon) * eig_sign
        graph_matrix = torch.matmul(eig_vec, torch.matmul(eig_val_smoothed.diag_embed(), eig_vec.transpose(-2, -1)))
    elif norm == 'Sym':
        deg = adj.sum(1)
        sym_norm = deg.pow(epsilon).unsqueeze(-1)
        graph_matrix = torch.matmul(sym_norm, sym_norm.transpose(0, 1)) * adj
    else:
        raise ValueError('Unknown norm called {}'.format(norm))

    identity = torch.eye(graph_matrix.shape[0], dtype=graph_matrix.dtype)
    bases = [identity.flatten(0)]

    graph_matrix_n = identity
    for shift in range(power):
        graph_matrix_n = torch.matmul(graph_matrix_n, graph_matrix)
        bases = bases + [graph_matrix_n.flatten(0)]

    bases = torch.stack(bases, dim=0).transpose(-2, -1).contiguous()

    full_one = torch.ones_like(graph_matrix, dtype=graph_matrix.dtype).nonzero(as_tuple=True)
    # print(full_one)
    new_g = dgl.graph(full_one)
    assert (new_g.num_nodes() == g.num_nodes())
    # new_g = DGLHeteroGraph(full_one, ['_U'], ['_E'])
    new_g.ndata['feat'] = g.ndata['feat']
    # new_g.ndata['_ID'] = g.ndata['_ID']
    new_g.edata['bases'] = bases
    if 'feat' in g.edata.keys():
        edge_attr = g.edata.pop('feat')
        # print(edge_attr)
        if len(edge_attr.shape) == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        edge_attr_dense = to_dense_adj(edge_idx, edge_attr=edge_attr).squeeze(0).view(-1, edge_attr.shape[-1])
        assert (len(edge_attr_dense.shape) == 2)
        assert (bases.shape[0] == edge_attr_dense.shape[0])
        new_g.edata['feat'] = edge_attr_dense
    # new_g.edata['_ID'] = g.edata['_ID']
    # print(new_g)
    return new_g
