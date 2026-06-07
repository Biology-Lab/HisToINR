import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse
from scanpy.get import obs_df
from scanpy.plotting import spatial
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from tqdm import tqdm
import os
import scanpy as sc
import anndata as ad
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from HisToINR.networks import *
from HisToINR.get_UNI_imageFeature import UNI_features

np.random.seed(1234)
slice_idx = [0, 1, 2,3,4,5]

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

        # HER2-positive breast cancer dataset  UNI Image Feature Extractor
        all_features = []
        for i, adata in enumerate(adata_st_list_raw):
            # step1: read image
            section_id = str(slice_idx[i]+1)
            img_path = os.path.join('Images/HER2-positive',
                                              f'B{section_id}.jpg')
            # step2: read spatial coord
            spatial = (adata.obsm['spatial']).astype(int)
            # step3: extract image feature
            image_feat_uni = UNI_features(img_path, spatial)
            # step4: collect image features of all slices
            all_features.append(image_feat_uni)
        combined_features = np.vstack(all_features)
        image_feat = torch.tensor(combined_features, dtype=torch.float32, requires_grad=True)
        image_feat_np = image_feat.detach().cpu().numpy()
        adata_st.obsm['image_feat_uni'] = image_feat_np
        adata_st.write('data/HER2-positive/B/adata_st_HER2_modified1.h5ad')

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

        # 用来存储每个切片的最高 ARI
        highest_ari = 0 # 初始化每个切片的最高 ARI

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

            if step % 1000 == 0:

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


                result = self.eval(adata_st_list_raw, save=False, output_path="./results_HER2")

                np.random.seed(self.seed)
                gm = GaussianMixture(n_components=4, covariance_type='tied',
                                     reg_covar=10e-4, init_params='kmeans')
                y = gm.fit_predict(self.latent, y=None)
                self.adata_st.obs["GM"] = [str(z) for z in y]
                result[0].obs["GM"] = self.adata_st.obs.loc[result[0].obs_names,]["GM"]

                # upload ground truth
                Ann_df = pd.read_csv(os.path.join(
                    'HER2-positiveB_annotation.txt'), sep='\t',
                    header=None, index_col=0)
                Ann_df.columns = ['Ground Truth']
                result[0].obs_names = result[0].obs_names
                result[0].obs['Ground Truth'] = Ann_df.loc[result[0].obs_names, 'Ground Truth']

                obs_df = result[0].obs.dropna()
                obs_df = obs_df[obs_df['Ground Truth'] != 'undetermined']
                ari = adjusted_rand_score(obs_df["GM"], obs_df['Ground Truth'])

                # paint
                coords = result[0].obsm['spatial']
                labels = result[0].obs["GM"]

                labels_cat = pd.Categorical(labels)
                categories = labels_cat.categories
                codes = labels_cat.codes

                cmap = plt.get_cmap("tab20")
                colors = cmap(np.linspace(0, 1, len(categories)))
                spot_colors = [colors[i] for i in codes]

                fig, ax = plt.subplots(figsize=(6, 6))

                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=spot_colors,
                    s=40,
                    linewidths=0,
                    alpha=0.9
                )

                handles = [
                    plt.Line2D([0], [0],
                               marker='o',
                               color='w',
                               markerfacecolor=colors[i],
                               markersize=8,
                               label=categories[i])
                    for i in range(len(categories))
                ]

                ax.legend(
                    handles=handles,
                    title="Cluster",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    frameon=False
                )

                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.axis('off')

                plt.title(f"A:ARI = {ari:.3f}", fontsize=14)

                plt.tight_layout()
                plt.savefig(f"A_ARI={ari:.3f}.pdf", dpi=600, bbox_inches="tight")
                plt.show()

                if ari > highest_ari:
                    highest_ari = ari
                # sc.pl.spatial(result[0], img_key="hires",
                #               color=["GM", "Ground Truth"], title=['Slice' + str(slice_idx[0]) + ' results ARI=' +
                #                                                    str(round(ari, 3)), 'Ground Truth'])
        print(highest_ari)

    def eval(self, adata_st_list_raw, save=False, output_path="./results"):
        self.net.eval()
        self.Z, self.beta, self.alpha, self.gamma = self.net.evaluate(
            self.A, self.coord, self.X, self.slice)

        # add learned representations to full ST adata object
        embeddings = self.Z.detach().cpu().numpy()
        cell_reps = pd.DataFrame(embeddings)
        cell_reps.index = self.adata_st.obs.index
        self.adata_st.obsm['latent'] = cell_reps.loc[self.adata_st.obs_names,].values
        self.latent = cell_reps.loc[self.adata_st.obs_names,].values
        if save == True:
            cell_reps.to_csv(os.path.join(output_path, "representation.csv"))

        # add deconvolution results to original anndata objects
        b = self.beta.detach().cpu().numpy()
        n_spots = 0
        adata_st_decon_list = []
        for i, adata_st_i in enumerate(adata_st_list_raw):
            adata_st_i.obs.index = adata_st_list_raw[i].obs.index + "-slice%d" % i
            decon_res = pd.DataFrame(b[n_spots:(n_spots + adata_st_list_raw[i].shape[0]), :],
                                     columns=self.celltypes)
            decon_res.index = adata_st_list_raw[i].obs.index
            adata_st_i.obs = adata_st_list_raw[i].obs.join(decon_res)
            n_spots += adata_st_i.shape[0]
            adata_st_decon_list.append(adata_st_i)

        return adata_st_decon_list