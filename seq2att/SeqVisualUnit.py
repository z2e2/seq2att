import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

font = {'family': 'sans-serif',
        'size'   : 12}
mpl.rc('font', **font)

LETTER_FP = FontProperties(family="monospace", weight="bold") 
globscale = 1.35
LETTERS = { "T" : TextPath((-0.305, 0), "T", size=1, prop=LETTER_FP),
            "G" : TextPath((-0.384, 0), "G", size=1, prop=LETTER_FP),
            "A" : TextPath((-0.35, 0), "A", size=1, prop=LETTER_FP),
            "C" : TextPath((-0.366, 0), "C", size=1, prop=LETTER_FP)}

COLOR_SCHEME = {'A': 'red', 
                'C': 'blue', 
                'G': 'orange', 
                'T': 'darkgreen'}

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

COLOR_MAP = ['red', 'orange', 'green', 'blue', 'gold', 
             'lightskyblue', 'brown', 'gray', 'pink',
             'yellow', 'purple', 'goldenrod', 'violet', 
             'burlywood', 'tan', 'chocolate', 
             'salmon', 'coral', 'moccasin', 'darkseagreen',
             'lime', 'teal', 'darkturquoise', 'navy',
             'rosybrown', 'navajowhite', 'khaki', 'deepskyblue',
             'indigo', 'magenta', 'darkred', 'saddlebrown', 
             'greenyellow', 'darkmagenta'
            ]

MARKER_MAP = ['o', '^', 'p', 'd', 's']

EPS = np.finfo(float).eps

class SeqVisualUnit:
    '''
    Object used to visualize sequences (embedding, attention and sequence logo)
    sequence logo plot was adapted from # source: https://stackoverflow.com/questions/42615527/sequence-logos-in-matplotlib-aligning-xticks?answertab=votes#tab-top
    '''
    def __init__(self, X, y, idx_to_label, taxa_label_list, prediction, attention_weights, sequence_embedding, output_dir):
        '''
        Initialize the visualization object
        '''
        self.X = X
        self.y = y
        self.unique_y = np.unique(y)
        self.idx_to_label = idx_to_label
        self.taxa = taxa_label_list
        self.idx_to_taxa = {}
        self.taxa_to_idx = {}
        self.unique_taxa_list = sorted(np.unique(taxa_label_list))
        self.output_dir = output_dir
        for idx, taxa in enumerate(self.unique_taxa_list):
            self.idx_to_taxa[idx] = taxa
            self.taxa_to_idx[taxa] = idx
        self.y_taxa = np.array([self.taxa_to_idx[taxa] for taxa in taxa_label_list])
        self.pred = np.argmax(prediction, axis=1)
        self.att = attention_weights
        self.emb = sequence_embedding
        self.emb_2d = None
        logging.info('Visualization object initialized: Read2Phenotype Accuracy: {:.4f}.'.format(accuracy_score(self.y, self.pred)))
        
    def plot_embedding(self):
        '''
        plot embedding in 2D.
        '''
        if len(self.unique_taxa_list) > len(COLOR_MAP) or self.unique_y.shape[0] > len(MARKER_MAP):
            logging.info('Visualization Failed: too many taxa or phenotype to be plotted, please cutomized your plot.')
            return
            
        if self.emb_2d is None:
            pca_model = PCA(n_components=2)
            pca_model = pca_model.fit(self.emb)
            self.emb_2d = pca_model.transform(self.emb)
        
        plt.figure(figsize=(15,10))
        for label_idx in range(self.unique_y.shape[0]):
            plt.scatter(self.emb[self.y == self.unique_y[label_idx],0], 
                        self.emb[self.y == self.unique_y[label_idx],1], 
                        s=50, alpha=0.5, marker=MARKER_MAP[label_idx], color='k', facecolors='none', label='in {}'.format(self.idx_to_label[label_idx])) 

        for idx, taxa in enumerate(self.unique_taxa_list):
            plt.scatter(self.emb[self.y_taxa == self.taxa_to_idx[taxa],0], 
                        self.emb[self.y_taxa == self.taxa_to_idx[taxa],1], 
                        s=15, alpha=0.5, marker='o', color=COLOR_MAP[idx], label=taxa) 
                           
        lgnd = plt.legend(loc="lower center", ncol=3)
        for idx, item in enumerate(lgnd.legendHandles):
            item._sizes = [100]
            item.set_alpha(1)

        plt.xlabel('Axis 1')
        plt.ylabel('Axis 2')
        plt.savefig('{}/embedding_visualization.pdf'.format(self.output_dir), format='pdf', dpi=300, bbox_inches='tight')       
        plt.show()
        
    def _get_sequencelogo(self, subset_seq, visualization_threshold=0.2):
        '''
        base probability for sequence logo 
        '''
        if subset_seq is None or subset_seq.shape[0] == 0:
            logging.error('Invalid input to compute sequence logo.')
            return None
        idx2base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        subset_seq_int = np.argmax(subset_seq, axis = -1)
        sequence_logo = np.zeros((4, subset_seq_int.shape[-1]))
        nt = np.array(range(0, subset_seq_int.shape[-1]))
        for i in range(subset_seq_int.shape[-1]):
            inds, counts = np.unique(subset_seq_int[:,i], return_counts=True)
            sequence_logo[inds, i] = counts
        sequence_logo = sequence_logo/np.sum(sequence_logo, axis = 0)
        return sequence_logo
    
    def _moving_average(self, x, w):
        '''
        Moving average of attention weights
        '''
        return np.convolve(x, np.ones(w), 'same') / w

    def _entropy(self, sequence_logo):
        '''
        Entropy per location
        '''
        H = np.zeros(sequence_logo.shape[1])
        for i in range(sequence_logo.shape[1]):
            probA=(sequence_logo[0, i] + EPS)
            probT=(sequence_logo[1, i] + EPS)
            probG=(sequence_logo[2, i] + EPS)
            probC=(sequence_logo[3, i] + EPS)
            H[i] = -1*(probA*np.log2(probA) + probT*np.log2(probT) + probG*np.log2(probG) + probC*np.log2(probC))
        return H
    
    def _letterAt(self, letter, x, y, yscale=1, ax=None):
        '''
        Draw letter at a location
        '''
        text = LETTERS[letter]

        t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
            mpl.transforms.Affine2D().translate(x,y) + ax.transData
        p = PathPatch( text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
        if ax != None:
            ax.add_artist(p)
        return p

    def _get_nt_scores(self, weblogo):
        '''
        get nucleotide score for sequence logo plot
        '''
        scores = []
        for i in range(weblogo.shape[1]):
            tmp = [('A', weblogo[0, i]), ('C', weblogo[1, i]), ('G', weblogo[2, i]), ('T', weblogo[3, i])]
            tmp.sort(key= lambda x: x[1])
            scores.append(tmp)
        return scores

    def plot_sequence_logo(self, ax, tmp_logo, xlabel, ylabel):
        '''
        plot sequence logo (nucleotide frequency scaled by entropy)
        '''
        ALL_SCORES2 = self._get_nt_scores(tmp_logo)
        all_scores = ALL_SCORES2
        x = 1
        maxi = 0
        for scores in all_scores:
            y = 0
            for base, score in scores:
                self._letterAt(base, x, y, score, ax)
                y += score
            x += 1
            maxi = max(maxi, y)
        maxi += maxi * 0.05
        ax.set_xticklabels([i + 1 for i in range(tmp_logo.shape[1])], rotation = 90)
        ax.tick_params(labelsize = 25)
        ax.get_yaxis().set_visible(True)
        ax.set_xticks(range(1,x))
        ax.set_xlim((0, x)) 
        ax.set_ylim((0, maxi))  
        ax.set_ylabel(ylabel, fontsize = 30) 
        if xlabel:
            ax.set_xlabel(xlabel, fontsize = 30) 
        plt.show()

    
    def get_att_sequencelogo(self, taxa_name, visualization_threshold=0.2):
        '''
        get sequence logo for attention interpretation
        '''
        taxa_id = self.taxa_to_idx[taxa_name]

        subset_seq = {}
        subset_att = {}
        subset_sequencelogo = {}
        for idx in range(self.unique_y.shape[0]):
            label = self.idx_to_label[self.unique_y[idx]]
            subset_seq[label] = self.X[(self.y_taxa == taxa_id) & (self.pred==idx), :, :]
            subset_att[label] = self.att[(self.y_taxa == taxa_id) & (self.pred==idx), :, :]
            subset_sequencelogo[label] = self._get_sequencelogo(subset_seq[label])

        class_to_visual = []
        for key in subset_seq:
            if subset_seq[key].shape[0] > self.X[self.y_taxa == taxa_id].shape[0] * visualization_threshold:
                class_to_visual.append(key)
        logging.info('Classes to be visualized: {}.'.format(', '.join(class_to_visual)))

        overall_seq = self.X[self.y_taxa == taxa_id, :, :]
        overall_att = np.mean(self.att[self.y_taxa == taxa_id, :, 0], axis = 0)
        overall_sequencelogo = self._get_sequencelogo(overall_seq)


        return subset_sequencelogo, subset_att, overall_sequencelogo, overall_att, class_to_visual
    
    def plot_attention(self, taxa_name):
        '''
        plot attention weights for sequences from a certain taxonomic unit.
        '''
        subset_sequencelogo, subset_att, overall_sequencelogo, overall_att, class_to_visual = self.get_att_sequencelogo(taxa_name, visualization_threshold=0.2)

        H = self._entropy(overall_sequencelogo)
        SEQ_LEN = H.shape[0]
        for idx, phenotype_label in enumerate(class_to_visual):
            logging.info('Attention Visualization: Plotting {} in {}.'.format(taxa_name, phenotype_label))

            tmp_att = np.mean(subset_att[phenotype_label][:, :, 0], axis = 0)    
            tmp_att = self._moving_average(tmp_att, 9)

            scaled_entropy = subset_sequencelogo[phenotype_label] * H

            num_data = np.concatenate((np.arange(SEQ_LEN).reshape(SEQ_LEN,1), tmp_att.reshape(SEQ_LEN,1)), axis = 1)

            fig = plt.figure(figsize=(50, 5))
            ax = fig.add_subplot(1,1,1)
            z = np.tile(np.repeat(tmp_att, SEQ_LEN), (10, 1))
            x, y = np.meshgrid(np.linspace(0.5, 100.5, 10000), np.linspace(0, 1.25, 10))
            c = ax.pcolormesh(x, y, z, cmap = 'pink', alpha = 0.8, vmin=np.min(z), vmax=np.max(z))
            fig.colorbar(c, ax = ax, pad = 0.01)
            self.plot_sequence_logo(ax, scaled_entropy, None, 'NT Sequence Logo')
            ax.set_ylim([0, 1.25])
            ax.set_xlim([0.5, 100.5])
            ax.set_yticks([0.0, 0.625, 1.25])
            ax.set_yticklabels([str(i) for i in [0.0, 0.625, 1.25]], fontsize = 30)
            fig.savefig('{}/attention_visualization-{}-{}.png'.format(self.output_dir, taxa_name, phenotype_label), format='png', dpi=150, bbox_inches='tight')
            plt.show()

        tmp_att = self._moving_average(overall_att, 9)
        scaled_entropy = overall_sequencelogo * H
        num_data = np.concatenate((np.arange(SEQ_LEN).reshape(SEQ_LEN,1), tmp_att.reshape(SEQ_LEN,1)), axis = 1)
        logging.info('Attention Visualization: Plotting {} from all.'.format(taxa_name))
        fig = plt.figure(figsize=(50, 5))
        ax = fig.add_subplot(1,1,1)
        z = np.tile(np.repeat(tmp_att, SEQ_LEN), (10, 1))
        x, y = np.meshgrid(np.linspace(0.5, 100.5, 10000), np.linspace(0, 1.25, 10))
        c = ax.pcolormesh(x, y, z, cmap = 'pink', alpha = 0.8, vmin=np.min(z), vmax=np.max(z))
        fig.colorbar(c, ax = ax, pad = 0.01)
        self.plot_sequence_logo(ax, scaled_entropy, None, 'NT Sequence Logo')
        ax.set_ylim([0, 1.25])
        ax.set_xlim([0.5, 100.5])
        ax.set_yticks([0.0, 0.625, 1.25])
        ax.set_yticklabels([str(i) for i in [0.0, 0.625, 1.25]], fontsize = 30)
        fig.savefig('{}/attention_visualization-{}-overall.png'.format(self.output_dir, taxa_name), format='png', dpi=150, bbox_inches='tight')
        plt.show()