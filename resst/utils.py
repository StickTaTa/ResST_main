import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
import sklearn


def count_nbr(target_cluster, cell_id, spatial, pred, radius):
    adj_2d = sklearn.metrics.pairwise_distances(spatial)
    cluster_num = dict()
    df = {'cell_id': cell_id, 'x': spatial[:, 0], "y": spatial[:, 1], "pred": pred}
    df = pd.DataFrame(data=df)
    df.index = df['cell_id']
    target_df = df[df["pred"] == target_cluster]
    row_index = 0
    num_nbr = []
    for index, row in target_df.iterrows():
        x = row["x"]
        y = row["y"]
        tmp_nbr = df[((df["x"] - x) ** 2 + (df["y"] - y) ** 2) <= (radius ** 2)]
        num_nbr.append(tmp_nbr.shape[0])
    return np.mean(num_nbr)


def search_radius(target_cluster, cell_id, spatial, pred, start, end, num_min=8, num_max=15, max_run=100):
    run = 0
    num_low = count_nbr(target_cluster, cell_id, spatial, pred, start)
    num_high = count_nbr(target_cluster, cell_id, spatial, pred, end)
    if num_min <= num_low <= num_max:
        print("recommended radius = ", str(start))
        return start
    elif num_min <= num_high <= num_max:
        print("recommended radius = ", str(end))
        return end
    elif num_low > num_max:
        print("Try smaller start.")
        return None
    elif num_high < num_min:
        print("Try bigger end.")
        return None
    while (num_low < num_min) and (num_high > num_min):
        run += 1
        print("Run " + str(run) + ": radius [" + str(start) + ", " + str(end) + "], num_nbr [" + str(
            num_low) + ", " + str(num_high) + "]")
        if run > max_run:
            print("Exact radius not found, closest values are:\n" + "radius=" + str(start) + ": " + "num_nbr=" + str(
                num_low) + "\nradius=" + str(end) + ": " + "num_nbr=" + str(num_high))
            return None
        mid = (start + end) / 2
        num_mid = count_nbr(target_cluster, cell_id, spatial, pred, mid)
        if num_min <= num_mid <= num_max:
            print("recommended radius = ", str(mid), "num_nbr=" + str(num_mid))
            return mid
        if num_mid < num_min:
            start = mid
            num_low = num_mid
        elif num_mid > num_max:
            end = mid
            num_high = num_mid


def find_neighbor_clusters(target_cluster, cell_id, x, y, pred, radius, ratio=1 / 2):
    cluster_num = dict()
    for i in pred:
        cluster_num[i] = cluster_num.get(i, 0) + 1
    df = {'cell_id': cell_id, 'x': x, "y": y, "pred": pred}
    df = pd.DataFrame(data=df)
    df.index = df['cell_id']
    target_df = df[df["pred"] == target_cluster]
    nbr_num = {}
    row_index = 0
    num_nbr = []
    for index, row in target_df.iterrows():
        x = row["x"]
        y = row["y"]
        tmp_nbr = df[((df["x"] - x) ** 2 + (df["y"] - y) ** 2) <= (radius ** 2)]
        # tmp_nbr=df[(df["x"]<x+radius) & (df["x"]>x-radius) & (df["y"]<y+radius) & (df["y"]>y-radius)]
        num_nbr.append(tmp_nbr.shape[0])
        for p in tmp_nbr["pred"]:
            nbr_num[p] = nbr_num.get(p, 0) + 1
    del nbr_num[target_cluster]
    nbr_num = [(k, v) for k, v in nbr_num.items() if v > (ratio * cluster_num[k])]
    nbr_num.sort(key=lambda x: -x[1])
    print("radius=", radius, "average number of neighbors for each spot is", np.mean(num_nbr))
    print(" Cluster", target_cluster, "has neighbors:")
    for t in nbr_num:
        print("Domain ", t[0], ": ", t[1])
    ret = [t[0] for t in nbr_num]
    if len(ret) == 0:
        print("No neighbor domain found, try bigger radius or smaller ratio.")
    else:
        return ret


def rank_genes_groups(input_adata, target_cluster, nbr_list, label_col, adj_nbr=True, log=False):
    if adj_nbr:
        nbr_list = nbr_list + [target_cluster]
        adata = input_adata[input_adata.obs[label_col].isin(nbr_list)]
    else:
        adata = input_adata.copy()
    adata.var_names_make_unique()
    adata.obs["target"] = ((adata.obs[label_col] == target_cluster) * 1).astype('category')
    sc.tl.rank_genes_groups(adata, groupby="target", reference="rest", n_genes=adata.shape[1], method='wilcoxon')
    pvals_adj = [i[0] for i in adata.uns['rank_genes_groups']["pvals_adj"]]
    genes = [i[1] for i in adata.uns['rank_genes_groups']["names"]]
    if issparse(adata.X):
        obs_tidy = pd.DataFrame(adata.X.A)
    else:
        obs_tidy = pd.DataFrame(adata.X)
    obs_tidy.index = adata.obs["target"].tolist()
    obs_tidy.columns = adata.var.index.tolist()
    obs_tidy = obs_tidy.loc[:, genes]
    # 1. compute mean value
    mean_obs = obs_tidy.groupby(level=0).mean()
    # 2. compute fraction of cells having value >0
    obs_bool = obs_tidy.astype(bool)
    fraction_obs = obs_bool.groupby(level=0).sum() / obs_bool.groupby(level=0).count()
    # compute fold change.
    if log:  # The adata already logged
        fold_change = np.exp((mean_obs.loc[1] - mean_obs.loc[0]).values)
    else:
        fold_change = (mean_obs.loc[1] / (mean_obs.loc[0] + 1e-9)).values
    df = {'genes': genes, 'in_group_fraction': fraction_obs.loc[1].tolist(),
          "out_group_fraction": fraction_obs.loc[0].tolist(),
          "in_out_group_ratio": (fraction_obs.loc[1] / fraction_obs.loc[0]).tolist(),
          "in_group_mean_exp": mean_obs.loc[1].tolist(), "out_group_mean_exp": mean_obs.loc[0].tolist(),
          "fold_change": fold_change.tolist(), "pvals_adj": pvals_adj}
    df = pd.DataFrame(data=df)
    return df
