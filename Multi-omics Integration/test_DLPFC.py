import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from HisToINR.extend import model
from marker_gene import *
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# slice_idx = [151507, 151508, 151509, 151510]
slice_idx = [151669,151670,151671,151672]
# slice_idx = [151673, 151674, 151675, 151676]

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

# add extend information: cell_count、maker_gene
df1 = pd.read_csv('matched_result_group2.csv')  # 包含# 三列：barcode, expr_chrM_ratio, cell_count
df1 = df1.set_index('barcode')
adata_st.obs['expr_chrM_ratio'] = df1['expr_chrM_ratio'].reindex(adata_st.obs.index)
adata_st.obs['cell_count'] = df1['cell_count'].reindex(adata_st.obs.index)
df = pd.read_excel("KRM_Layer_Markers.xlsx")
layers = [1,2,3,4,5,6,"WM"]
marker_table = {}
for _, row in df.iterrows():
    gene = row["Gene"].upper()
    b_data = row[layers].values.astype(np.float32)
    marker_table[gene] = torch.tensor(b_data)
m = build_marker_embeddings(adata_st,marker_table)

adata_st.obsm["m"] = m.detach().cpu().numpy()

model = model.Model(adata_st_list_raw, adata_st, adata_basis)

model.train()

save_path = ""
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)