import numpy as np
import pandas as pd
import igraph as ig
from collections import Counter
from IPython.display import Image

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

## plotting utility function
def plot_path(sg, path, dct_processname, fn="_temp.png"):
    sg.vs['vertex_size'] = 2
    sg.vs['vertex_label_size'] = 1
    for i in path:
        sg.vs[i]['vertex_size'] = 1
        sg.vs[i]['vertex_label_size'] = 20
    ig.plot(sg, target=fn, bbox=(1000,700), layout=sg.layout_reingold_tilford(), margin=50, 
            vertex_size=sg.vs['vertex_size'], 
            vertex_label=[dct_processname.get(x,'') for x in sg.vs['uid']],
            vertex_label_size=sg.vs['vertex_label_size'], 
            edge_color='lightgrey', edge_arrow_size=0)    
    return Image(fn)
