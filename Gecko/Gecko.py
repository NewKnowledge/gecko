# The following may be needed on some (potentially noninteractive) environments
import matplotlib
#matplotlib.use('Agg')
import os
os.environ["THEANO_FLAGS"]="mode=FAST_RUN,device=gpu,floatX=float32"
import networkx as nx
from time import time
import matplotlib.pyplot as plt

from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
# graph embedding methods to cycle through
from gem.embedding.gf import GraphFactorization
from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

class Gecko:
    def __init__(self):
        # Initialize set of possible models
        # see "Graph Embedding Techniques, Applications, and Performance: A Survey" by 
        # Goyal and Ferrera (2017) for a taxonomy of graph embedding methods
        models = []
        # Presently all methods are "factorization based methods"
        models.append(GraphFactorization(d=2, max_iter=100000, eta=1*10**-4, regu=1.0))
        models.append(HOPE(d=4, beta=0.01))
        models.append(LaplacianEigenmaps(d=2))
        models.append(LocallyLinearEmbedding(d=2))
        # The following "random walk based" and "deep learning based" methods will be enabled in the future
        # models.append(node2vec(d=2, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1))
        # models.append(SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50, xeta=0.01,n_batch=500,
        #                modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'],
        #                weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5']))
        self.models = models
    
    # Convert to graph embedding, then reconstruct nodes, to measure suitability of particular method
    # on similar graphs. Return most "suitable" model
    def GraphReconstruction(self,G,verbose=True,visualize=True):
        maxMAP=0
        for embedding in self.models:
            if(verbose):
                print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))

            # Learn embedding - accepts a networkx graph or file with edge list
            t1 = time()
            Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
            if(verbose):
                print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
            # Evaluate on graph reconstruction
            MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
            #---------------------------------------------------------------------------------
            print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
            #---------------------------------------------------------------------------------
            # Visualize
            if(visualize):
                viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
                # plt.show() # one can display using 'TkAgg' matplotlib backend
                plt.savefig("embedding_"+embedding._method_name) # saving figure with 'Agg' matplotlib backend

            # keep track of the best embedding so far
            if(maxMAP<MAP):
                bestEmbedding = embedding
                maxMAP=MAP
            
        return bestEmbedding

    def CommunityDetection(self,G,embedding,n_clusters=2,visualize=True):
        kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(embedding.get_embedding())
        X_=embedding.get_embedding()

        # Visualize using tsne if needed
        if(X_.shape[1]>2):
            X_= TSNE(n_components=2).fit_transform(X_)
        pos ={}
        n_nodes = X_.shape[0]
        for i in range(n_nodes):
            pos[i] = X_[i,:]
        if(visualize):
            nx.draw_networkx(G,pos,node_color=kmeans.labels_,node_size=300,alpha=0.5,arrows=False,font_size=12)
            plt.title('Community Detection using Graph Embedding '+embedding._method_name)
            plt.show() # one can display using 'TkAgg' matplotlib backend
            plt.savefig("community_detection_"+embedding._method_name) # saving figure with 'Agg' matplotlib backend

        return kmeans
if __name__=='__main__':
    # if you are using in pip script form, you will need these imports
    # from Gecko import Gecko
    # from gem.utils import graph_util

    # GRAPH RECONSTRUCTION
    # File that contains the edges. Format: source target
    edge_f = 'scripts/data/karate.edgelist'
    # Specify whether the edges are directed
    isDirected = True
    # Load graph
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
    G = G.to_directed()
    embedding_generator = Gecko()
    bestEmbedding = embedding_generator.GraphReconstruction(G=G,visualize=False)
    print("DEBUG::The best embedding found is")
    print(bestEmbedding)
    # advance users can also define their own embeddings directly, as done in the __init__ method above

    # Community Detection/ Node Clustering using Graph Embeddings
    communities = embedding_generator.CommunityDetection(G=G,embedding=bestEmbedding)

