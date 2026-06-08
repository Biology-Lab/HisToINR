import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
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

slice_idx = [151507, 151508, 151509, 151510]
# slice_idx = [151669, 151670, 151671, 151672]
# slice_idx = [151673, 151674, 151675, 151676]

adata_st_list_raw0 = ad.read_h5ad('151507.h5ad')
adata_st_list_raw1 = ad.read_h5ad('151508.h5ad')
adata_st_list_raw2 = ad.read_h5ad('151509.h5ad')
adata_st_list_raw3 = ad.read_h5ad('151510.h5ad')
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

adata_st = ad.read_h5ad('adata_st_DLPFC_modified1_group1.h5ad')
adata_basis = ad.read_h5ad('adata_basis_DLPFC_group1.h5ad')

model = model.Model(adata_st_list_raw, adata_st, adata_basis)

model.train()

save_path = ""
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)

from sklearn.mixture import GaussianMixture

np.random.seed(1234)
gm = GaussianMixture(n_components=7, covariance_type='tied',
                     reg_covar=10e-4, init_params='kmeans')
y = gm.fit_predict(model.adata_st.obsm['latent'], y=None)
model.adata_st.obs["GM"] = y
model.adata_st.obs["GM"].to_csv(os.path.join(save_path, "clustering_result.csv"))

order = [0, 1, 2, 3, 4, 5, 6]  # reordering cluster labels

model.adata_st.obs["Cluster"] = [order[label] for label in model.adata_st.obs["GM"].values]

for i in range(len(result)):
    result[i].obs["GM"] = model.adata_st.obs.loc[result[i].obs_names,]["GM"]
    result[i].obs["Cluster"] = model.adata_st.obs.loc[result[i].obs_names,]["Cluster"]

# ============================================================
# cell-type deconvolution strategy
# ============================================================

""" Align spots between HisToINR deconvolution results and layer annotations."""
def align_spots_histoinr(adata, truth_df):
    # Remove slice suffix from spot barcodes
    adata_barcodes = adata.obs.index.str.replace(r"-slice\d+$", "", regex=True)
    # Find common spot barcodes
    common_barcodes = adata_barcodes.intersection(truth_df.index)

    mask = adata_barcodes.isin(common_barcodes)
    adata_aligned = adata[mask].copy()

    # Reorder ground truth to match AnnData spot order
    truth_aligned = truth_df.loc[adata_barcodes[mask].values].copy()

    # sanity check
    assert all(
        adata_aligned.obs.index.str.replace(r"-slice\d+$", "", regex=True)
        == truth_aligned.index
    ), "❌ spot alignment failed!"

    return adata_aligned, truth_aligned


# ============================================================
# Cell Type → Cortical Layer Mapping
# ===
celltype_layer_map = {
    "Ex_10_L2_4": ["Layer_2", "Layer_3", "Layer_4"],

    "Ex_3_L4_5": ["Layer_4"],
    "Ex_6_L4_6": ["Layer_4"],
    "Ex_7_L4_6": ["Layer_4"],

    "Ex_2_L5": ["Layer_5"],
    "Ex_5_L5": ["Layer_5"],

    "Ex_1_L5_6": ["Layer_6"],
    "Ex_4_L_6": ["Layer_6"],
    "Ex_8_L5_6": ["Layer_6"],
    "Ex_9_L5_6": ["Layer_6"],
}

def filter_layers(truth_df, valid_layers=None):
    if valid_layers is None:
        valid_layers = ["Layer_2", "Layer_3", "Layer_4", "Layer_5", "Layer_6"]

    mask = truth_df["layer"].isin(valid_layers)
    return truth_df[mask], mask


import numpy as np
from sklearn.metrics import roc_auc_score

# ============================================================
# Cell-Type-Specific AUC Evaluation
# ============================================================

def compute_celltype_auc_scientific(adata, truth_df, celltype_layer_map):
    """
    adata: HisToINR deconv AnnData
    truth_df: dataframe with spot layer annotation
    celltype_layer_map: dict, celltype -> list of target layers
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np

    auc_dict = {}

    for celltype, target_layers in celltype_layer_map.items():
        if celltype not in adata.obs.columns:
            continue

        # Predicted proportion of the cell type
        beta = adata.obs[celltype].values

        # Evaluate each associated layer separately
        for layer in target_layers:
            # Prediction score
            y_score = beta

            # Binary label:
            # 1 = spot belongs to target layer
            # 0 = spot belongs to other layers
            y_true = (truth_df["layer"] == layer).astype(int).values

            # calculate AUC
            if len(np.unique(y_true)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y_true, y_score)

            # key: celltype_layer
            auc_dict[f"{celltype}"] = auc

    return auc_dict


def run_histoinr_celltype_auc(result, annotation_root):
    all_auc = []

    for i, adata_st_i in enumerate(result):
        section_id = str(slice_idx[i])
        # Load layer annotations
        truth_path = os.path.join('DLPFC_annotations', section_id + '_truth.txt')
        truth_df = pd.read_csv(truth_path, sep="\t", header=None, names=["barcode", "layer"])
        truth_df = truth_df.set_index("barcode")

        # 1️⃣ align spot
        adata_aligned, truth_aligned = align_spots_histoinr(adata_st_i, truth_df)

        # 2️⃣ filter layer L1 and WM
        truth_filtered, mask = filter_layers(truth_aligned)
        adata_filtered = adata_aligned[mask].copy()

        # 3️⃣ calculate celltype AUC
        auc_dict = compute_celltype_auc_scientific(adata_filtered, truth_filtered, celltype_layer_map)

        auc_series = pd.Series(auc_dict, name=section_id)
        all_auc.append(auc_series)


    auc_df = pd.DataFrame(all_auc)
    return auc_df
# ============================================================
# Run Evaluation
# ============================================================

auc_df = run_histoinr_celltype_auc(
    result=result,
    annotation_root="DLPFC_annotations"
)

print(auc_df)
print("\nMean AUC per celltype:")
print(auc_df.mean())

# ============================================================
# Visualization of Deconvolution Results
# ============================================================
for i, adata_st_i in enumerate(result):

    slice_id = str(slice_idx[i])
    celltypes = [ct for ct in auc_df.columns]

    titles = []
    for ct in celltypes:

        auc = auc_df.loc[slice_id, ct] if slice_id in auc_df.index else np.nan
        if np.isnan(auc):
            titles.append(f"{ct}\nAUC: NA")
        else:
            titles.append(f"{ct}\nAUC: {auc:.3f}")

    sc.pl.spatial(
        adata_st_i,
        img_key="lowres",
        color=celltypes,
        size=1.0,
        title=titles,
        ncols=4,
        cmap="viridis",
        frameon=False,
        show=False
    )

    plt.savefig(
        f"{slice_id}_raw.pdf",
        dpi=600,
        bbox_inches="tight"
    )
    plt.close()
