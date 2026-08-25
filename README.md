# Matching_Experiments

Experiments using our matching algorithm found at: https://github.com/AMS-Hippo/tree_matching over several datasets

### ACME4

* ```process_eda.ipynb```: Build process dataframe from "process_uber_summary" and "sigma_labels" tables. EDA, classification, feature importance and visualization
* ```process_embedding.ipynb```: Embed ACME4 processes using top features and save.
* ```process_trees_eda.ipynb```: Build process trees from process_uber_summary table. EDA, classification and clustering experiments, visualization. Also contains experiments and plots for our WAW 2026 paper (also on arXiv at: arxiv.org/abs/2602.04694)
* ```process_trees_matching.ipynb```:  Matching algorithm: compare all "baduser" trees (templates) with all other trees, find best matches. Also consider various process embeddings and clusterings. Matching algorithm: root process classification using multiple templates.
* ```process_trees_motifs.ipynb```: experimental

### DMBD

* ```clean_trees.ipynb```: Build the process trees.
* ```DMBD_matching.ipynb```: Several experiments with our matching algorithm.

### COMISET

* TBD

### Python files

* ```process_trees.py```: utility functions to extract features, build trees, etc.
* ```fast_match.py```: Matching algorithm
* ```igraph_io.py``` and ```tree_data.py```: utility functions for the Matching algorithm
* ```gw_matcher.py```: old (slow) version of the Matching algorithm, for sanity check
