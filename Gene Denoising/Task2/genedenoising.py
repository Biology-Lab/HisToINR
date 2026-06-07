import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.io
import matplotlib.pyplot as plt
import sys
from HisToINR import model
import warnings
import cv2
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sympy import false

warnings.filterwarnings("ignore")

import torch
import random

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# slice_idx = [151507,151508,151509,151510]
slice_idx = [151669,151670,151671,151672]
# slice_idx = [151673,151674,151675,151676]
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

adata_st_list_raw0 = ad.read_h5ad('151669.h5ad')
adata_st_list_raw1 = ad.read_h5ad('151670.h5ad')
adata_st_list_raw2 = ad.read_h5ad('151671.h5ad')
adata_st_list_raw3 = ad.read_h5ad('151672.h5ad')
adata_st_list_raw = []
adata_st_list_raw.append(adata_st_list_raw0)
adata_st_list_raw.append(adata_st_list_raw1)
adata_st_list_raw.append(adata_st_list_raw2)
adata_st_list_raw.append(adata_st_list_raw3)

celltype_list_use = ['Astros_1', 'Astros_2', 'Astros_3',
                     'Endo', 'Micro/Macro',
                     'Oligos_1', 'Oligos_2', 'Oligos_3',
                     'Ex_1_L5_6', 'Ex_2_L5', 'Ex_3_L4_5',
                     'Ex_4_L_6', 'Ex_5_L5',
                     'Ex_6_L4_6', 'Ex_7_L4_6', 'Ex_8_L5_6',
                     'Ex_9_L5_6', 'Ex_10_L2_4']

adata_st = ad.read_h5ad('adata_st_DLPFC_modified1_group2.h5ad')
adata_basis = ad.read_h5ad('adata_basis_DLPFC_group2.h5ad')

model = model2.Model(adata_st_list_raw, adata_st, adata_basis)
model.train()

save_path = ""
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)

# the third-party algorithms:KMeans and Agglomerative
cluster = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)
# kmeans = KMeans(n_clusters=7,random_state=0)

for i in range(len(result)):
    # cluster
    X_denoised = result[i].obsm["X_pred"]
    result[i].obs['hc'] = cluster.fit_predict(X_denoised)
    # evaluate
    section_id = str(slice_idx[i])
    Ann_df = pd.read_csv(os.path.join(
        'DLPFC_annotations',
        section_id+'_truth.txt'), sep='\t',
        header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    result[i].obs_names = [z[:-7] for z in result[i].obs_names]
    result[i].obs['Ground Truth'] = Ann_df.loc[result[i].obs_names, 'Ground Truth']
    obs_df = result[i].obs.dropna()
    ari = adjusted_rand_score(obs_df["hc"], obs_df['Ground Truth'])

    print("ARI:", ari)
    # visualization
    sc.pl.spatial(result[i], img_key="hires",
                  color=["hc"], title=['Slice' + str(slice_idx[i]) + ' results ARI=' +
                                       str(round(ari, 3))],frameon=false,show=false)

    plt.savefig(
        f"{section_id}_{str(round(ari, 3))}.pdf",
        dpi=600,
        bbox_inches="tight"
    )
    plt.close()

