import torch
import pickle
import torch.utils.data
import time
import numpy as np
import os
import os.path as osp
import csv
import dgl
from tqdm import tqdm
import sys
from scipy import sparse as sp
import numpy as np
sys.path.append("..")
from utils.basis_transform import basis_transform
from dgl.data.utils import load_graphs, save_graphs
from utils.config import get_config_from_json


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs, norm, epsilon, power, identity):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        self.pre_processed_file_path = osp.join(data_dir, '%s_dgl_data_processed' % self.split)
        self.labels_file_path = osp.join(data_dir, '%s_labels' % self.split)

        with open(data_dir + "/%s.pickle" % self.split, "rb") as f:
            # with open('data/ZINC.pkl', "rb") as f:

            self.data = pickle.load(f)

        # loading the sampled indices from file ./zinc_molecules/<split>.index
        with open(data_dir + "/%s.index" % self.split, "r") as f:
            data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
            self.data = [self.data[i] for i in data_idx[0]]

        assert len(self.data) == num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"

        """
        data is a list of Molecule dict objects with following attributes

          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """

        self.norm = norm
        self.epsilon = epsilon
        self.power = power
        self.identity = identity

        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()

    def _prepare(self):
        if os.path.exists(self.pre_processed_file_path):
            print('Already Done, Loading...')
            self.graph_lists, _ = load_graphs(self.pre_processed_file_path)
            self.graph_labels = torch.load(self.labels_file_path)

        else:
            print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))

            trans_start = time.time()
            for step, molecule in enumerate(tqdm(self.data, desc="Pre-processing")):
                node_features = molecule['atom_type'].long()

                adj = molecule['bond_type']
                edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list

                edge_idxs_in_adj = edge_list.split(1, dim=1)
                edge_features = adj[edge_idxs_in_adj].reshape(-1).long()

                # Create the DGL Graph
                g = dgl.DGLGraph()
                g.add_nodes(molecule['num_atom'])
                g.ndata['feat'] = node_features

                for src, dst in edge_list:
                    g.add_edges(src.item(), dst.item())
                g.edata['feat'] = edge_features

                g = dgl.remove_self_loop(g)
                g = dgl.add_self_loop(g)
                # assert (g.edata['feat'].shape[0] == _num_edges + g.num_nodes())
                g = basis_transform(g, norm=self.norm, epsilon=self.epsilon, power=self.power, identity=self.identity)

                self.graph_lists.append(g)
                self.graph_labels.append(molecule['logP_SA_cycle_normalized'])

            total_time = time.time() - trans_start
            print('S^{\epsilon} transformation total and avg time:', total_time, total_time / len(self.data))
            print('Saving...')
            save_graphs(self.pre_processed_file_path, self.graph_lists)
            torch.save(self.graph_labels, self.labels_file_path)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name

        self.num_atom_type = 28  # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4  # known meta-info about the zinc dataset; can be calculated as well

        data_dir = './data/molecules'

        config = get_config_from_json("./ZINC.json")
        norm = config.norm
        epsilon = config.epsilon
        power = config.power
        identity = config.get('identity', 1)
        print('Basis configurations: norm: {}, epsilon: {}, power: {}, identity: {}'.format(norm, epsilon, power, identity))

        self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000, norm=norm, epsilon=epsilon, power=power, identity=identity)
        self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000, norm=norm, epsilon=epsilon, power=power, identity=identity)
        self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000, norm=norm, epsilon=epsilon, power=power, identity=identity)
        print("Time taken: {:.4f}s".format(time.time() - t0))


class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name, norm='none', pos_enc_dim=0, verbose=True):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/molecules/'
        with open(data_dir + name + '.pkl', "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.stack(labels)
        # print(labels)
        # labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels


dataset = MoleculeDatasetDGL(name='ZINC')
num_atom_type = 28
num_bond_type = 4
start = time.time()
with open('data/molecules/ZINC.pkl','wb') as f:
        pickle.dump([dataset.train,dataset.val,dataset.test,num_atom_type,num_bond_type],f)
print('Time (sec):',time.time() - start)
