import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import os

import scipy.spatial.distance
import sklearn
import stlearn
from scipy.spatial import distance

from model_ST_utils import trainer, priori_cluster, plot_map
from get_adata import get_data, refine
from preprocess import get_enhance_feature

data_name = 'V1_Breast_Cancer_Block_A_Section_1'
generated_data_path = 'generated_data'
model_path = os.path.join('model/', data_name)
# embedding_data_path = os.path.join('Embedding_data/', data_name)
result_path = os.path.join('results/', data_name)
n_clusters = 19

if not os.path.exists(os.path.join(generated_data_path, data_name) + '/adata_index.h5'):
    adata = get_data(data_path='dataset', data_name=data_name,
                     generated_data_path=generated_data_path,
                     cnnType='ResNet50')

    adata = get_enhance_feature(adata)

    adata.write(os.path.join(generated_data_path, data_name) + '/adata_index.h5')
else:
    adata = sc.read_h5ad(os.path.join(generated_data_path, data_name) + '/adata_index.h5')

# if not os.path.exists(embedding_data_path):
#     os.makedirs(embedding_data_path)
# if not os.path.exists(model_path):
#     os.makedirs(model_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

# adata = model_on_ST(adata, n_clusters, model_path, embedding_data_path, result_path, num_epoch=1500)
adata = trainer(adata, save_path=result_path, data_name=data_name)

cluster_adata = anndata.AnnData(adata.obsm["embed"])
cluster_adata.obs_names = adata.obs_names
sc.pp.neighbors(cluster_adata, n_neighbors=15)

res = priori_cluster(cluster_adata, n_domains=n_clusters)

sc.tl.leiden(cluster_adata, key_added="pred", resolution=res, random_state=0)
adata.obs['pred'] = cluster_adata.obs['pred']
######### Strengthen the distribution of points in the model
adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
refined_pred = refine(sample_id=adata.obs.index.tolist(),
                      pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
adata.obs["refine_pred"] = refined_pred

plot_map(adata, save_path=result_path, data_name=data_name)

adata.write(result_path + '/adata_main.h5')

# ====================================================================================================


# adj_2d = sklearn.metrics.pairwise_distances(adata.obsm['spatial'])
# refined_pred = refine(sample_id=adata.obs.index.tolist(),
#                       pred=adata.obs["kmeans"].tolist(), dis=adj_2d, shape="hexagon")
# adata.obs['refine'] = refined_pred
# sc.pl.spatial(adata, img_key=None, color='refine', show=True,
#               legend_loc='right margin', legend_fontsize='x-large', size=1.6)
#
# sc.pl.spatial(adata, img_key=None, color='kmeans', show=True,
#               legend_loc='right margin', legend_fontsize='x-large', size=1.6)
#
# adata.write(result_path + '/adata_main.h5')
