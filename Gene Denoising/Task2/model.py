import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse
from numba.core.cgutils import printf
from scanpy.plotting import spatial
from tqdm import tqdm
import os
import scanpy as sc
import anndata as ad
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from HisToINR.networks import *
from HisToINR.get_UNI_imageFeature import UNI_features
from sklearn.metrics import mean_squared_error, mean_tweedie_deviance
import scipy.sparse as sp


np.random.seed(1234)

class Model():

    def __init__(self, adata_st_list_raw, adata_st, adata_basis,
                 hidden_dims=[512, 128],
                 n_heads=1,
                 slice_emb_dim=16,
                 coef_fe=0.1,
                 training_steps=11,
                 lr=0.001,
                 seed=112,
                 distribution="Poisson"
                 ):

        import random
        self.seed = seed
        self.adata_basis = adata_basis
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.lr = lr
        self.training_steps = training_steps

        self.adata_st = adata_st
        self.celltypes = list(adata_basis.obs.index)

        # add device
        self.device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.hidden_dims = [adata_st.shape[1]] + hidden_dims
        self.n_celltype = adata_basis.shape[0]
        self.n_slices = len(sorted(set(adata_st.obs["slice"].values)))

        # build model
        if distribution == "Poisson":
            self.net = DeconvNet(hidden_dims=self.hidden_dims,
                                 n_celltypes=self.n_celltype,
                                 n_slices=self.n_slices,
                                 n_heads=n_heads,
                                 slice_emb_dim=slice_emb_dim,
                                 adj_dim=torch.from_numpy(
                                     np.array(adata_st.obsm["graph"])
                                 ).float().to(self.device).shape[1],
                                 coef_fe=coef_fe,
                                 ).to(self.device)
        else:  # Negative Binomial distribution
            self.net = DeconvNet_NB(hidden_dims=self.hidden_dims,
                                    n_celltypes=self.n_celltype,
                                    n_slices=self.n_slices,
                                    n_heads=n_heads,
                                    slice_emb_dim=slice_emb_dim,
                                    coef_fe=coef_fe,
                                    ).to(self.device)

        self.optimizer = optim.Adamax(list(self.net.parameters()), lr=lr)


        # read data
        if scipy.sparse.issparse(adata_st.X):
            self.X = torch.from_numpy(adata_st.X.toarray()).float().to(self.device)

        else:
            self.X = torch.from_numpy(adata_st.X).float().to(self.device)
        self.A = torch.from_numpy(np.array(adata_st.obsm["graph"])).float().to(self.device)
        self.Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(self.device)
        self.lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(self.device)
        self.slice = torch.from_numpy(np.array(adata_st.obs["slice"].values)).long().to(self.device)
        self.basis = torch.from_numpy(np.array(adata_basis.X)).float().to(self.device)
        self.coord = torch.from_numpy(np.array(adata_st.obsm['3D_coor'])).float().to(self.device)
        self.image_feat_uni = torch.from_numpy(np.array(adata_st.obsm["image_feat_uni"])).float().to(self.device)

    def train(self, report_loss=True, step_interval=500):
        self.net.train()

        for step in tqdm(range(self.net.training_steps)):
            loss, recon, denoise, Z_, ind_min, ind_max = self.net(coord=self.coord,
                                                                  image_feat_uni = self.image_feat_uni,
                                                                  adj_matrix=self.A,
                                                                  node_feats=self.X,
                                                                  count_matrix=self.Y,
                                                                  library_size=self.lY,
                                                                  slice_label=self.slice,
                                                                  basis=self.basis,
                                                                  step=step)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self, adata_st_list_raw, save=False, output_path="./results"):
        self.net.eval()
        self.Z, self.mid_fea,self.beta, self.alpha, self.gamma = self.net.evaluate(
            self.A, self.coord, self.X, self.slice)

        # add learned representations to full ST adata object
        embeddings = self.Z.detach().cpu().numpy()
        cell_reps = pd.DataFrame(embeddings)
        cell_reps.index = self.adata_st.obs.index
        self.adata_st.obsm['latent'] = cell_reps.loc[self.adata_st.obs_names,].values
        self.latent = cell_reps.loc[self.adata_st.obs_names,].values
        if save == True:
            cell_reps.to_csv(os.path.join(output_path, "representation.csv"))

        # add gene imputation results to original anndata objects
        g = self.mid_fea.detach().cpu().numpy()
        n_spots = 0
        adata_st_recov_list = []
        for i, adata_st_i in enumerate(adata_st_list_raw):
            adata_st_i.obs.index = adata_st_list_raw[i].obs.index + "-slice%d" % i
            recov_res = g[n_spots:(n_spots + adata_st_list_raw[i].shape[0])]
            adata_st_i.obsm['X_pred'] = recov_res
            adata_st_i.uns['pred_gene_names'] = self.adata_st.var_names
            n_spots += adata_st_i.shape[0]
            adata_st_recov_list.append(adata_st_i)

        return adata_st_recov_list


