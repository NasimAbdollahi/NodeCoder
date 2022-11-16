import pandas as pd
import numpy as np
import gzip
import Bio.PDB
import numpy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyvis.network import Network


class Visualize_Prediction(object):

    def __init__(self, args):
        self.args = args
        self.visulize_protein_prediction()

    def calc_residue_dist(self, residue_one, residue_two):
        """Returns the C-alpha distance between two residues"""
        diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
        return np.sqrt(numpy.sum(diff_vector * diff_vector))

    def calc_contacts(self, chain_one, chain_two):
        """Returns a matrix of C-alpha distances between two chains"""
        answer = np.zeros((len(chain_one), len(chain_two)), 'float')
        contacts = []
        for row, residue_one in enumerate(chain_one):
            for col, residue_two in enumerate(chain_two):
                distance = self.calc_residue_dist(residue_one, residue_two)
                if distance <= 10:
                    contacts.append((col, row))
        return contacts

    def convert_fraction_to_color(self, fraction):
        assert(0 <= fraction <= 1.0)
        color_0 = (255, 255, 255)   # Start Color
        color_1 = (255, 120, 0)     # End Color
        color_n = tuple([int(color_0[x] + ((color_1[x] - color_0[x]) * fraction)) for x in range(3)])
        color = '#%02x%02x%02x' % color_n
        return color

    def visulize_protein_prediction(self):
        df = pd.read_csv(self.args.protein_prediction_fileName)
        if self.args.structure_file.split('.')[-1] == 'gz':
            protein_structure = gzip.open(self.args.structure_file, 'rt')
            structure = Bio.PDB.PDBParser().get_structure('XX', protein_structure)
        else:
            protein_structure = self.args.structure_file
            structure = Bio.PDB.PDBParser().get_structure(self.args.protein_ID, protein_structure)
        model = structure[0]
        contacts = self.calc_contacts(model["A"], model["A"])

        for j in self.args.target_name:
            target_name = j.split('_')[1]
            for t in ['Target', 'Prediction Probability']:
                nt = Network(notebook=True)
                nodes = set()
                for i, row in df.iterrows():
                    f = row[target_name + ' ' + t]
                    nt.add_node(i, size=50, label=f'x{i+1}', color=self.convert_fraction_to_color(f))
                    nodes.add(i)
                for x, y in contacts:
                    if x in nodes and y in nodes:
                        nt.add_edge(x, y)
                nt.toggle_physics(True)
                nt.show(self.args.path_protein_results + self.args.protein_ID + '_' + target_name + '_' + t + '.html')
                nt.toggle_physics(True)

        #################################################

        for j in self.args.target_name:
            target_name = j.split('_')[1]
            color_map_G_t, color_map_G_p, prob_list = [], [], []
            G = nx.Graph()
            nodes = set()
            for i, row in df.iterrows():
                task = row[f"{target_name} Target"]
                prob = row[f"{target_name} Prediction Probability"]
                G.add_node(i, size=50, label=f'x{i+1}', color=self.convert_fraction_to_color(task))
                nodes.add(i)
                color_map_G_t.append(self.convert_fraction_to_color(task))
                color_map_G_p.append(self.convert_fraction_to_color(prob))
                prob_list.append(prob)
            for x, y in contacts:
                if x in nodes and y in nodes:
                    G.add_edge(x, y)

            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
            axs[0] = plt.subplot(211)
            pos = nx.spring_layout(G)
            nx.draw(G, pos=pos, node_size=30, with_labels=False, node_color=color_map_G_t, edge_color='#CCCCCC')#,alpha=0.5)
            axs[0].collections[0].set_edgecolor('#808080')
            nx.draw_networkx_edges(G, pos=pos, width=0.3, edge_color='#CCCCCC')
            axs[1] = plt.subplot(212)
            nx.draw(G, pos=pos, node_size=22, with_labels=False, node_color=color_map_G_p, edge_color='#CCCCCC')
            axs[1].collections[0].set_edgecolor('#808080')

            cmap = mpl.colors.ListedColormap([self.convert_fraction_to_color(0.1), self.convert_fraction_to_color(0.3),
                                              self.convert_fraction_to_color(0.5), self.convert_fraction_to_color(0.7),
                                              self.convert_fraction_to_color(0.9)])
            bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            fig.subplots_adjust(top=0.9)
            cbar_ax = fig.add_axes([0.25, 0.08, 0.45, 0.031])
            fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                         label='prediction probability', orientation='horizontal', cax=cbar_ax)
            axs[0].set_title('Target', fontweight="bold", size=10)
            axs[1].set_title('Prediction', fontweight="bold", size=10)
            plt.savefig(self.args.path_protein_results + self.args.protein_ID + '_' + target_name + '_2D.png', dpi=200)