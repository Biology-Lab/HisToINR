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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

slice_idx = [0,1,2,3,4,5]

# Patient: A~G A~D: 6 slices E~G: 3 slices
adata_st_list_raw0 = ad.read_h5ad('data/HER2-positive/B/adata_st_list_0.h5ad')
adata_st_list_raw1 = ad.read_h5ad('data/HER2-positive/B/adata_st_list_1.h5ad')
adata_st_list_raw2 = ad.read_h5ad('data/HER2-positive/B/adata_st_list_2.h5ad')
adata_st_list_raw3 = ad.read_h5ad('data/HER2-positive/B/adata_st_list_3.h5ad')
adata_st_list_raw4 = ad.read_h5ad('data/HER2-positive/B/adata_st_list_4.h5ad')
adata_st_list_raw5 = ad.read_h5ad('data/HER2-positive/B/adata_st_list_5.h5ad')


adata_st_list_raw = []
adata_st_list_raw.append(adata_st_list_raw0)
adata_st_list_raw.append(adata_st_list_raw1)
adata_st_list_raw.append(adata_st_list_raw2)
adata_st_list_raw.append(adata_st_list_raw3)
adata_st_list_raw.append(adata_st_list_raw4)
adata_st_list_raw.append(adata_st_list_raw5)


adata_st = ad.read_h5ad('/data/HER2-positive/B/adata_st_HER2.h5ad')
adata_basis = ad.read_h5ad('data/HER2-positive/B/adata_basis_HER2.h5ad')

model = model.Model(adata_st_list_raw, adata_st, adata_basis)

model.train()

save_path = ""
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)

