import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
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


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load raw dataset
slice_idx = [151673, 151674, 151675, 151676]

adata_st_list_raw0 = ad.read_h5ad('/home/guodingfei/MyProject/STINR-main/STINR-main/adata_st_list_raw0.h5ad')
adata_st_list_raw1 = ad.read_h5ad('/home/guodingfei/MyProject/STINR-main/STINR-main/adata_st_list_raw1.h5ad')
adata_st_list_raw2 = ad.read_h5ad('/home/guodingfei/MyProject/STINR-main/STINR-main/adata_st_list_raw2.h5ad')
adata_st_list_raw3 = ad.read_h5ad('/home/guodingfei/MyProject/STINR-main/STINR-main/adata_st_list_raw3.h5ad')
adata_st_list_raw = []
adata_st_list_raw.append(adata_st_list_raw0)
adata_st_list_raw.append(adata_st_list_raw1)
adata_st_list_raw.append(adata_st_list_raw2)
adata_st_list_raw.append(adata_st_list_raw3)

# deconvolution celltype_list
celltype_list_use = ['Astros_1', 'Astros_2', 'Astros_3',
                     'Endo', 'Micro/Macro',
                     'Oligos_1', 'Oligos_2', 'Oligos_3',
                     'Ex_1_L5_6', 'Ex_2_L5', 'Ex_3_L4_5',
                     'Ex_4_L_6', 'Ex_5_L5',
                     'Ex_6_L4_6', 'Ex_7_L4_6', 'Ex_8_L5_6',
                     'Ex_9_L5_6', 'Ex_10_L2_4']

# load the merged dataset and reference gene expression matrix
adata_st = ad.read_h5ad('adata_st_DLPFC_modified1.h5ad')
adata_basis = ad.read_h5ad('adata_basis_DLPFC.h5ad')


model = model.Model(adata_st_list_raw, adata_st, adata_basis)

model.train()

save_path = ""
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)