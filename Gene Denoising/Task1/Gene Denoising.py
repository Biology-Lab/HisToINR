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

slice_idx = [0,1,2]

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

adata_st_list_raw0 = ad.read_h5ad('data/MBC/adata_st_section1.h5ad')
adata_st_list_raw1 = ad.read_h5ad('data/MBC/adata_st_section2.h5ad')
adata_st_list_raw2 = ad.read_h5ad('data/MBC/adata_st_section3.h5ad')
adata_st_list_raw = []
adata_st_list_raw.append(adata_st_list_raw0)
adata_st_list_raw.append(adata_st_list_raw1)
adata_st_list_raw.append(adata_st_list_raw2)

adata_st = ad.read_h5ad('data/MBC/adata_st_MBC_modified1.h5ad') # crop_size = 64
adata_basis = ad.read_h5ad('data/MBC/adata_basis_MBC.h5ad')

model = model.Model(adata_st_list_raw, adata_st, adata_basis)
model.train()

save_path = ""
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)
