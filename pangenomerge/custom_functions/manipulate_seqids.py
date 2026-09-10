import networkx as nx
import pandas as pd

# then replace individual run seqID with seqID from run of all data
def indSID_to_allSID(G, gid_map):

    for node in G.nodes():
        node_SIDs = G.nodes[node].get("seqIDs", "")  
        if node_SIDs:
            updated_SIDs = [
                gid_map.get(sid.strip(), "error")  # put 'error' if not found
                for sid in node_SIDs
            ]
            G.nodes[node]["seqIDs"] = updated_SIDs
    return G

# get individual seqIDs as keys, and the cluster each seqID belongs to as its value
def get_seqIDs_in_nodes(G):  

    dictionary = {}

    for node in G.nodes():
        seq_ids = set(G.nodes[node].get("seqIDs", "error"))

        for SID in seq_ids:
            dictionary[SID] = node

    return dictionary

# The clustering metrics that consume these dicts (RI/ARI/MI/AMI, pcARI/pcNMI,
# acARI/acNMI, homogeneity/completeness) live in
# pangenomerge.custom_functions.graph_similarity -- one implementation, shared by
# --mode test and the thesis benchmarking scripts.

# flatten seqID/cluster dictionaries for input into clustering metrics
def dict_to_2d_array(d):
    row_keys = []
    row_values = []

    for key, value in d.items():
        if isinstance(value, (list, tuple)):
            for v in value:
                row_keys.append(key)
                row_values.append(v)
        else:
            row_keys.append(key)
            row_values.append(value)

    return pd.DataFrame([row_keys, row_values])
