import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from pathlib import Path
from torch.autograd import Variable
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from tqdm import tqdm

from .model import ResST, AdversarialNetwork
import resst.graph_utils as graph_utils


def optimize_cluster(adata, resolution: list = list(np.arange(0.1, 2.5, 0.01))):
    """

    :param adata:
    :param resolution:
    :return:
    """
    scores = []
    for r in resolution:
        sc.tl.leiden(adata, resolution=r)
        s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
        scores.append(s)
    cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
    best_idx = np.argmax(cl_opt_df["score"])
    res = cl_opt_df.iloc[best_idx, 0]
    print("Best resolution: ", res)
    return res


def priori_cluster(adata, n_domains=15, cluster_type='leiden', increment=0.01):
    """
    If we have the number of clusters
    :param adata: [AnnaData matrix]
    :param n_domains: numbers of clusters[int]
    :param cluster_type: methods for clustering
    :param increment: increment[int]
    :return: resolution[int]
    """
    if cluster_type == 'leiden':
        print('Please waiting, calculating best resolution...')
        for res in sorted(list(np.arange(0.1, 2.5, increment)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            if count_unique_leiden == n_domains:
                break
        print("Best resolution: ", res)
    elif cluster_type == 'louvain':
        print('Please waiting, calculating best resolution...')
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            if count_unique_louvain == n_domains:
                break
        print("Best resolution: ", res)
    return res


def trainer(adata, data_name, save_path, domains=None,
            pre_epochs=1000,
            epochs=500,
            min_cells=3,
            pca_n_comps=50,
            linear_encoder_hidden=[32, 20],
            linear_decoder_hidden=[32, 60],
            conv_hidden=[32, 8],
            p_drop=0.01, dec_cluster_n=20, lr=5e-4,
            weight_decay=1e-4, grad_down=5, store=True, use_model=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    # linear_encoder_hidden = [128, 128]
    # linear_decoder_hidden = [128, 128]
    # conv_hidden = [128, 32]

    # linear_encoder_hidden = [50, 20]
    # linear_decoder_hidden = [50, 60]
    # conv_hidden = [32, 8]

    # linear_encoder_hidden = [32, 20]
    # linear_decoder_hidden = [32, 60]
    # conv_hidden = [32, 8]
    #
    # p_drop = 0.01
    # dec_cluster_n = 20
    # lr = 5e-4
    # weight_decay = 1e-4
    # grad_down = 5

    if domains is not None:
        domains = torch.from_numpy(domains).to(device)
    else:
        domains = domains

    graph_dict = graph_utils.graph(adata.obsm['spatial'], distType="BallTree", k=12, rad_cutoff=150).main()

    adata.X = adata.obsm['enhance_gene_data'].astype(float)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.log1p(adata_X)
    adata_X = sc.pp.scale(adata_X)
    concat_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    # concat_X = adata.X

    save_path_model = Path(os.path.join(save_path, "Model", data_name))
    save_path_model.mkdir(parents=True, exist_ok=True)

    if use_model:
        model = torch.load(os.path.join(save_path_model, f'{data_name}_model.pt'))
    else:

        model = ResST(
            input_dim=concat_X.shape[1],
            linear_encoder_hidden=linear_encoder_hidden,
            linear_decoder_hidden=linear_decoder_hidden,
            conv_hidden=conv_hidden,
            p_drop=p_drop,
            dec_cluster_n=dec_cluster_n
        )
        if domains is not None:
            model = AdversarialNetwork(model=model)
        else:
            model = model

        optimizer = torch.optim.Adam(params=list(model.parameters()), lr=lr, weight_decay=weight_decay)
        data = torch.FloatTensor(concat_X.copy()).to(device)
        adj = graph_dict['adj_norm'].to(device)
        model.to(device)

        # 预训练
        with tqdm(total=int(pre_epochs),
                  desc="Model is pretraining...",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(pre_epochs):
                inputs_corr = masking_noise(data)
                inputs_coor = inputs_corr.to(device)
                model.train()
                optimizer.zero_grad()
                if domains is None:
                    z, mu, logvar, de_feat, out_q, feat_x, gnn_z = model(Variable(inputs_coor), adj)
                    preds = model.dc(z)
                else:
                    z, mu, logvar, de_feat, _, feat_x, gnn_z, domain_pred = model(
                        Variable(inputs_coor), adj)
                    preds = model.model.dc(z)

                loss = model.loss(
                    decoded=de_feat,
                    x=data,
                    preds=preds,
                    labels=graph_dict['adj_label'].to(device),
                    mu=mu,
                    logvar=logvar,
                    n_nodes=data.shape[0],
                    norm=graph_dict['norm_value'],
                    mask=graph_dict['adj_label'].to(device),
                    mse_weight=10,
                    bce_kld_weight=0.1
                )
                if domains is not None:
                    source = -4 * nn.functional.cross_entropy(domain_pred, domains, reduction='none')
                    target = -nn.functional.nll_loss(
                        torch.log(torch.clamp(1. - nn.functional.softmax(domain_pred, dim=1) + 1e-6, max=1.)),
                        domains, reduction='none')
                    source_loss = source * torch.ones_like(source)
                    target_loss = target * torch.ones_like(target)
                    domain_loss = source_loss + target_loss
                    loss += domain_loss.mean()
                else:
                    loss = loss
                # print(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_down)
                optimizer.step()
                pbar.update(1)
        # 聚类初始化
        with torch.no_grad():
            model.eval()
            pre_z, _, _, _, _, _, _ = model(data, adj)
            pre_z = pre_z.cpu().detach().numpy()
            cluster_method = KMeans(n_clusters=dec_cluster_n, n_init=dec_cluster_n * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            if domains is None:
                model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(device)
            else:
                model.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(device)

        # 正式训练
        with tqdm(total=int(pre_epochs),
                  desc="Model is final training...",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(epochs):

                with torch.no_grad():
                    model.eval()
                    if domains is None:
                        z, _, _, _, q, _, _ = model(data, adj)
                    else:
                        z, _, _, _, q, _, _, _ = model(data, adj)
                    z = z.cpu().detach().numpy()
                    q = q.cpu().detach().numpy()
                q = model.target_distribution(torch.Tensor(q).clone().detach())
                y_pred = q.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if epoch > 0 and delta_label < 0.001:
                    print('delta_label {:.4}'.format(delta_label), '< tol', 0.001)
                    print('Reached tolerance threshold. Stopping training.')
                    break
                model.train()
                optimizer.zero_grad()
                inputs_coor = data.to(device)
                if domains is None:
                    z, mu, logvar, de_feat, out_q, feat_x, gnn_z = model(Variable(inputs_coor), adj)
                    preds = model.dc(z)
                else:
                    z, mu, logvar, de_feat, out_q, feat_x, gnn_z, domain_pred = model(Variable(inputs_coor), adj)
                    preds = model.model.dc(z)
                    loss_function = nn.CrossEntropyLoss()
                    domain_loss = loss_function(domain_pred, domains)
                loss_resst = model.loss(
                    decoded=de_feat,
                    x=data,
                    preds=preds,
                    labels=graph_dict['adj_label'].to(device),
                    mu=mu,
                    logvar=logvar,
                    n_nodes=data.shape[0],
                    norm=graph_dict['norm_value'],
                    mask=graph_dict['adj_label'].to(device),
                    mse_weight=10,
                    bce_kld_weight=0.1
                )
                loss_kl = F.kl_div(out_q.log(), q.to(device))
                if domains is None:
                    loss = 100 * loss_kl + loss_resst
                else:
                    loss = 100 * loss_kl + loss_resst + domain_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_down)
                optimizer.step()
                pbar.update(1)

        if store:
            torch.save(model, os.path.join(save_path_model, f'{data_name}_model.pt'))

    with torch.no_grad():
        model.eval()
        z, mu, logvar, de_feat, out_q, feat_x, gnn_z = model(data, adj)
        resst_embed = z.cpu().detach().numpy()
    print('resst training has been Done! the embeddings has been stored adata.obsm["embed"].')
    adata.obsm["embed"] = resst_embed
    # cluster_labels, score = Kmeans_cluster(resst_embed, n_clusters)
    # adata.obs['kmeans'] = cluster_labels

    return adata


def plot_map(adata, save_path, data_name, img_key=None,
             color='refine_pred',
             show=False,
             legend_loc='right margin',
             legend_fontsize='x-large',
             size=1.6,
             dpi=300):
    sc.pl.spatial(adata, img_key=img_key, color=color, show=show,
                  legend_loc=legend_loc, legend_fontsize=legend_fontsize, size=size)
    save_path_figure = Path(os.path.join(save_path, "Figure", data_name))
    save_path_figure.mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path_figure, f'{data_name}_umap.tif'),
                bbox_inches='tight', dpi=dpi)


def masking_noise(data, frac=0.01):
    """
    151673: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand < frac] = 0
    return data_noise
