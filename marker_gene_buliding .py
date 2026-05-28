import os
import pandas as pd
import torch
import numpy as np
import anndata as ad

'''Load marker table and build lookup table: create a table of [gene, 1, 2, 3, 4, 5, 6, WM]'''
df = pd.read_excel("KRM_Layer_Markers.xlsx")
layers = [1,2,3,4,5,6,"WM"] # Cortical layer indices plus white matter
marker_table = {} # Dictionary to store marker vector for each gene

# Iterate through each row in the marker Excel file
for _, row in df.iterrows():
    gene = row["Gene"].upper()
    b_data = row[layers].values.astype(np.float32)
    marker_table[gene] = torch.tensor(b_data)

'''Return a vector m[i] which is the average of marker vectors across all detected genes (present in marker_table) for the i-th spot'''
def build_marker_embeddings(adata, marker_table):
    n_spots = adata.n_obs
    marker_dim = 7
    m = torch.zeros((n_spots, marker_dim))
    for i, spot_genes in enumerate(adata.obs_names):
        # Extract expression vector for the current spot (convert to dense array and flatten)
        expressed = adata[i].X.toarray().flatten()
        indices = expressed.nonzero()[0] # Get indices of genes with non-zero expression
        genes = adata.var_names[indices] # Retrieve gene names for those indices
        vectors = []
        for g in genes:
            if g in marker_table:
                vectors.append(marker_table[g])
        if len(vectors) > 0:
            m[i] = torch.stack(vectors).mean(dim=0)

    return m






