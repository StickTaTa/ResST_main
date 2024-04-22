import stlearn as st
import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from anndata import AnnData
from sklearn.decomposition import PCA
from tqdm import tqdm
import random
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

from .read_adata_utils import *


# def read_stereoSeq(path,
#                    bin_size=100,
#                    is_sparse=True,
#                    library_id=None,
#                    scale=None,
#                    quality="hires",
#                    spot_diameter_fullres=1,
#                    background_color="white", ):
#     from scipy import sparse
#     count = pd.read_csv(os.path.join(path, "count.txt"), sep='\t', comment='#', header=0)
#     count.dropna(inplace=True)
#     if "MIDCounts" in count.columns:
#         count.rename(columns={"MIDCounts": "UMICount"}, inplace=True)
#     count['x1'] = (count['x'] / bin_size).astype(np.int32)
#     count['y1'] = (count['y'] / bin_size).astype(np.int32)
#     count['pos'] = count['x1'].astype(str) + "-" + count['y1'].astype(str)
#     bin_data = count.groupby(['pos', 'geneID'])['UMICount'].sum()
#     cells = set(x[0] for x in bin_data.index)
#     genes = set(x[1] for x in bin_data.index)
#     cellsdic = dict(zip(cells, range(0, len(cells))))
#     genesdic = dict(zip(genes, range(0, len(genes))))
#     rows = [cellsdic[x[0]] for x in bin_data.index]
#     cols = [genesdic[x[1]] for x in bin_data.index]
#     exp_matrix = sparse.csr_matrix((bin_data.values, (rows, cols))) if is_sparse else \
#         sparse.csr_matrix((bin_data.values, (rows, cols))).toarray()
#     obs = pd.DataFrame(index=cells)
#     var = pd.DataFrame(index=genes)
#     adata = AnnData(X=exp_matrix, obs=obs, var=var)
#     pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int)
#     adata.obsm['spatial'] = pos
#
#     if scale == None:
#         max_coor = np.max(adata.obsm["spatial"])
#         scale = 20 / max_coor
#
#     adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
#     adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale
#
#     # Create image
#     max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
#     max_size = int(max_size + 0.1 * max_size)
#     if background_color == "black":
#         image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
#     else:
#         image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
#     imgarr = np.array(image)
#
#     if library_id is None:
#         library_id = "StereoSeq"
#
#     adata.uns["spatial"] = {}
#     adata.uns["spatial"][library_id] = {}
#     adata.uns["spatial"][library_id]["images"] = {}
#     adata.uns["spatial"][library_id]["images"][quality] = imgarr
#     adata.uns["spatial"][library_id]["use_quality"] = quality
#     adata.uns["spatial"][library_id]["scalefactors"] = {}
#     adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
#     adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres
#
#     return adata


class image_feature:
    def __init__(
            self,
            adata,
            pca_components=50,
            cnnType='ResNet50',
            verbose=False,
            seeds=88,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnnType = cnnType

    def load_cnn_model(
            self,
    ):

        if self.cnnType == 'ResNet50':
            cnn_pretrained_model = models.resnet50(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Resnet152':
            cnn_pretrained_model = models.resnet152(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vgg19':
            cnn_pretrained_model = models.vgg19(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vgg16':
            cnn_pretrained_model = models.vgg16(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'DenseNet121':
            cnn_pretrained_model = models.densenet121(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Inception_v3':
            cnn_pretrained_model = models.inception_v3(pretrained=True)
            cnn_pretrained_model.to(self.device)
        else:
            raise ValueError(
                f"""\
                        {self.cnnType} is not a valid type.
                        """)
        return cnn_pretrained_model

    def extract_image_feat(
            self,
    ):

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225]),
                          transforms.RandomAutocontrast(),
                          transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                          transforms.RandomInvert(),
                          transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                          transforms.RandomSolarize(random.uniform(0, 1)),
                          transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2),
                                                  shear=(-0.3, 0.3, -0.3, 0.3)),
                          transforms.RandomErasing()
                          ]
        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize(mean=[0.54, 0.51, 0.68],
        #                   std =[0.25, 0.21, 0.16])]
        img_to_tensor = transforms.Compose(transform_list)

        feat_df = pd.DataFrame()
        model = self.load_cnn_model()
        # model.fc = torch.nn.LeakyReLU(0.1)
        model.eval()

        if "slices_path" not in self.adata.obs.keys():
            raise ValueError("Please run the function image_crop first")

        with tqdm(total=len(self.adata),
                  desc="Extract image feature",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]", ) as pbar:
            for spot, slice_path in self.adata.obs['slices_path'].items():
                spot_slice = Image.open(slice_path)
                spot_slice = spot_slice.resize((224, 224))
                spot_slice = np.asarray(spot_slice, dtype="int32")
                spot_slice = spot_slice.astype(np.float32)
                tensor = img_to_tensor(spot_slice)
                tensor = tensor.resize_(1, 3, 224, 224)
                tensor = tensor.to(self.device)
                result = model(Variable(tensor))
                result_npy = result.data.cpu().numpy().ravel()
                feat_df[spot] = result_npy
                feat_df = feat_df.copy()
                pbar.update(1)
        self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()
        if self.verbose:
            print("The image feature is added to adata.obsm['image_feat'] !")
        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feat_df.transpose().to_numpy())
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())
        if self.verbose:
            print("The pca result of image feature is added to adata.obsm['image_feat_pca'] !")
        return self.adata


def image_crop(
        adata,
        save_path,
        library_id=None,
        crop_size=50,
        target_size=224,
        verbose=False,
):
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)
    tile_names = []

    with tqdm(total=len(adata),
              desc="Tiling image",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            tile.thumbnail((target_size, target_size), Image.ANTIALIAS)  #####
            tile.resize((target_size, target_size))  ######
            tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(
                        str(imagecol), str(imagerow)))
            tile.save(out_tile, "PNG")
            pbar.update(1)

    adata.obs["slices_path"] = tile_names
    if verbose:
        print("The slice path of image feature is added to adata.obs['slices_path'] !")
    return adata


def get_data(data_path, data_name, generated_data_path, cnnType, pca_n_comps=200, platform="Visium", verbose=False):
    """
    读取不同平台的空转数据，默认为Visium,针对10X以及ST数据会进行图片处理
    :param generated_data_path:
    :param cnnType:
    :param pca_n_comps:
    :param data_path:
    :param data_name:
    :param platform:
    :param verbose:
    :return: adata
    """
    assert platform in ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
    if platform in ['Visium', 'ST']:
        if platform == 'Visium':
            adata = st.Read10X(os.path.join(data_path, data_name))
        else:
            adata = st.ReadOldST(os.path.join(data_path, data_name))
        sava_path_image_crop = Path(os.path.join(generated_data_path, data_name, 'Image_crop'))
        sava_path_image_crop.mkdir(parents=True, exist_ok=True)
        adata = image_crop(adata, save_path=sava_path_image_crop)
        adata = image_feature(adata, pca_components=pca_n_comps, cnnType=cnnType).extract_image_feat()

    elif platform == 'MERFISH':
        adata = read_merfish(os.path.join(data_path, data_name))
    elif platform == 'slideSeq':
        adata = read_SlideSeq(os.path.join(data_path, data_name))
    elif platform == 'seqFish':
        adata = read_seqfish(os.path.join(data_path, data_name))
    elif platform == 'stereoSeq':
        adata = read_stereoSeq(os.path.join(data_path, data_name))
    else:
        raise ValueError(
            f"""\
                                     {platform!r} does not support.
                                            """)
    if verbose:
        save_data_path = Path(os.path.join(generated_data_path, data_name))
        save_data_path.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(os.path.join(save_data_path, f'{data_name}.h5'), compression="gzip")

    return adata


def combine_adata(adata_1, adata_2):
    adata_1.obs['batch_name'] = '0'
    adata_2.obs['batch_name'] = '1'
    adata_1.obs['batch_name'] = adata_1.obs['batch_name'].astype('category')
    adata_2.obs['batch_name'] = adata_2.obs['batch_name'].astype('category')
    adata = AnnData.concatenate(adata_1, adata_2)
    domains = np.array(
        pd.Categorical(
            adata.obs['batch_name'],
            categories=np.unique(adata.obs['batch_name']), ).codes,
        dtype=np.int64,
    )
    return adata, domains
