import torch
import dgl
from torch_geometric.utils import to_dense_adj

# from utils.dense import to_dense_adj

torch.set_printoptions(
    precision=6,
    sci_mode=False
)


def basis_transform(g,
                  norm,
                  power,
                  epsilon,
                  identity):
    (e_idx0, e_idx1) = g.edges()
    edge_idx = torch.stack([e_idx0, e_idx1], dim=0)
    adj_i = to_dense_adj(edge_idx).squeeze(0)  # Graphs may have only one node.
    adj_wi = adj_i - torch.eye(adj_i.shape[0], dtype=adj_i.dtype) * (1 - identity)
    # dense_edge_idx = dense_edge_idx.double()

    eig_val_adj, eig_vec_adj = torch.linalg.eigh(adj_i - torch.eye(adj_i.shape[0], dtype=adj_i.dtype))
    eig_val_adj = torch.where(eig_val_adj.abs() <= 1e-6, torch.zeros_like(eig_val_adj), eig_val_adj)
    print(g.num_nodes(), '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(eig_val_adj)

    deg = adj_wi.sum(1)
    for eps in epsilon:
        if isinstance(eps, str):
            e = eps.split('d')
            eps = float(e[0]) / float(e[1])

        print('====', eps, '====')
        sym_norm = deg.pow(eps).unsqueeze(-1)
        graph_matrix = torch.matmul(sym_norm, sym_norm.transpose(0, 1)) * adj_wi  # Self-loop required !
        eig_val, eig_vec = torch.linalg.eigh(graph_matrix)
        eig_val = torch.where(eig_val.abs() <= 1e-6, torch.zeros_like(eig_val), eig_val)
        print((eig_val + 1e-5) / (eig_val_adj + 1e-5))

    full_one = torch.ones_like(graph_matrix, dtype=graph_matrix.dtype).nonzero(as_tuple=True)
    return dgl.graph(full_one)


# def basis_transform(g,
#                   norm,
#                   power,
#                   epsilon,
#                   identity):
#     (e_idx0, e_idx1) = g.edges()
#     edge_idx = torch.stack([e_idx0, e_idx1], dim=0)
#     adj_i = to_dense_adj(edge_idx).squeeze(0)  # Graphs may have only one node.
#     adj_wi = adj_i - torch.eye(adj_i.shape[0], dtype=adj_i.dtype) * (1 - identity)
#     # dense_edge_idx = dense_edge_idx.double()
#
#     eig_val_adj, eig_vec_adj = torch.linalg.eigh(adj_i - torch.eye(adj_i.shape[0], dtype=adj_i.dtype))
#     eig_val_adj = torch.where(eig_val_adj.abs() < 1e-6, torch.zeros_like(eig_val_adj), eig_val_adj)
#     print(g.num_nodes(), '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     lis = []
#     for i in range(1, g.num_nodes() + 1):
#         lis.append(i)
#     print(lis)
#     print(eig_val_adj / eig_val_adj[-1])
#
#     deg = adj_wi.sum(1)
#     for eps in epsilon:
#         print('====', eps, '====')
#         sym_norm = deg.pow(eps).unsqueeze(-1)
#         graph_matrix = torch.matmul(sym_norm, sym_norm.transpose(0, 1)) * adj_wi  # Self-loop required !
#         eig_val, eig_vec = torch.linalg.eigh(graph_matrix)
#         eig_val = torch.where(eig_val.abs() < 1e-6, torch.zeros_like(eig_val), eig_val)
#         print(eig_val / eig_val[-1])
#
#     full_one = torch.ones_like(graph_matrix, dtype=graph_matrix.dtype).nonzero(as_tuple=True)
#     return dgl.graph(full_one)
