import numpy as np
import pandas as pd
import igraph as ig
from collections import Counter
from IPython.display import Image
import matplotlib.pyplot as plt
import needleman_wunsch_tree as nwt ## slower, more flexibility for scoring functions

def extract_process_trees(process, uid='pid_hash', parent_uid='parent_pid_hash', extra=None, min_tree_size=3, max_tree_size=100):
    """
    Build process trees given uid's and parent uid's. 

    Args:
        process: dataframe with (at least) uid and parent_uid columns
        uid, parent_uid: name of those columns in the dataframe
        extra: dictionary of extra columns to extract from the dataframe
        min_tree_size, max_tree_size: int

    Returns:
        igraph.clustering.VertexClustering
    """
    ## keep required columns
    df = process.drop_duplicates()
    
    ## parents and children
    child = set(df[uid])
    parent = set(df[parent_uid]) ## Can be 'None', we'll delete later
    
    ## node dictionaries
    nodes = parent.union(child)
    nodes_dict = {v:k for k,v in enumerate(nodes)}
    inv_nodes_dict = {k:v for k,v in enumerate(nodes)}

    ## build directed graph from edgelist
    child = [nodes_dict[x] for x in df[uid]]
    parent = [nodes_dict[x] for x in df[parent_uid]]
    edges = np.array([parent,child]).T
    G = ig.Graph.TupleList(edges, directed=True)
    G = G.simplify() ## there are self-edges
    G.vs['uid'] = [inv_nodes_dict[int(x)] for x in G.vs['name']]

    ## find all connected components a.k.a. process trees (with one exception)
    Trees = G.connected_components(mode="weak")
    G.vs['tree'] = Trees.membership

    ## drop trees of size < min_tree_size and non-tree(s)
    _dct = dict(enumerate(Trees.sizes()))
    G.vs['tree_size'] = [_dct[i] for i in G.vs['tree']]
    roots = np.where(np.array(G.degree(mode='in'))==0)
    non_tree = set(np.array(G.vs['tree'])).difference(set(np.array(G.vs['tree'])[roots]))
    G.delete_vertices([v for v in G.vs if (v['tree_size']<min_tree_size or v['tree_size']>max_tree_size or v['tree'] in non_tree)])

    ## re-compute 
    Trees = G.connected_components(mode="weak")
    G.vs['tree'] = Trees.membership
    del G.vs['name']

    ## add some node features
    if extra:
        for i, (feature,name) in enumerate(extra.items()):
            user_dict = dict(zip(df[uid],df[feature]))
            G.vs[name] = [user_dict.get(x, '') for x in G.vs['uid']]    
    return Trees


def acme_list_process_subtrees(G, dct_username, dct_label, process_name='process', min_tree_size=3):
    """
    ACME4 specific function - list all subtrees of a process tree where the root process is named
    """
    L = []

    for v in G.vs:
        process = v[process_name]
        if G.degree(v, mode='out')>0 and process != '' and process != 'unknown': ## pick non-leaf nodes with some process name
            V, l, p = G.bfs(v.index)
            nodes = len(V)
            splits = sum([x>1 for x in list(Counter(np.array(p)[np.array(p)>=0]).values())])
            if nodes<min_tree_size: 
                continue
            leaves = nodes - len(set(np.array(p)[np.array(p)>=0]))
            layers = len(l)-1

            ## acme specific
            b3 = len([dct_username.get(x,'') for x in G.vs[V]['uid'] if x is not None and 'baduser3' in x])
            b9 = len([dct_username.get(x,'') for x in G.vs[V]['uid'] if x is not None and 'baduser9' in x])
            b25 = len([dct_username.get(x,'') for x in G.vs[V]['uid'] if x is not None and 'baduser25' in x])
            red = sum([dct_label.get(x,0) for x in G.vs[V]['uid']])
            
            x = [v.index, process, nodes, layers, leaves, b3, b9, b25, red]
            L.append(x)

    return pd.DataFrame(L, columns=['root','process','nodes','layers','leaves','bad3','bad9','bad25','redteam'])

## utility function - extract subtree from VertexClustering object
def acme_get_bfs_subtree(Trees, tree_id, root_id):
    """
    ACME4 specific function - extract a specific subtree 
    """    
    ## get subtree
    g = Trees.subgraph(tree_id)

    ## bfs ordering
    nodes, _ , parent = g.bfs(root_id)
    
    ## save vertices in new graph
    G = ig.Graph(directed=True)
    names = [str(g.vs['uid'][i]) for i in nodes] ## in BFS order
    G.add_vertices(names)
    
    ## save attributes in same order    
    for att in g.vs.attribute_names():
        G.vs[att] = [g.vs[att][i] for i in nodes]
    
    ## add edges
    E = [(str(g.vs['uid'][parent[i]]),str(g.vs['uid'][i])) for i in nodes if parent[i]>=0]
    G.add_edges(E)    

    del G.vs['name']
    
    return G

def run_redblack_match(TreeData, _label, weight, dct_label, min_red_nodes=1, verbose=False):
    best_score = 0
    for red in np.where(_label==True)[0]:
        for black in np.where(_label==False)[0]:
            match = nwt.align_trees_algorithm1(TreeData[red], TreeData[black], w=weight)
            if match.score > best_score:
                red_nodes = sum([dct_label[TreeData[red].label[x[0]]] for x in match.path_internal])
                if red_nodes >= min_red_nodes: 
                    best_score = match.score
                    best_pair = (red, black)
                    if verbose:
                        print('Best score:',best_score)
    return best_pair

## plotting utility function
def plot_path(sg, path, margin=[0,0]):

    titles = ["Redteam path","Non-redteam path"]
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    for i in [0,1]:
        sg[i].vs['label'] = sg[i].vs['process']
        sg[i].vs['size'] = 1
        sg[i].vs['label_size'] = 0
        for j in [x[i] for x in path]:
            sg[i].vs[j]['size'] = 0
            sg[i].vs[j]['label_size'] = 15
        ig.plot(sg[i], target=ax[i], layout=sg[i].layout_reingold_tilford(),
                edge_color='lightgrey', edge_arrow_size=0)    
        ax[i].set_title(titles[i], fontsize=18)
        ax[i].invert_yaxis()
        ax[i].margins(x=margin[i])
