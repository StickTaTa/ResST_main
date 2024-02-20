# ResST_main

![本地路径](fig/Figure_1.png)

# Overview
Spatial transcriptomics (ST) technologies allow for comprehensive characterization of gene expression patterns in the context of tissue microenvironment. However, accurately identifying domains with spatial coherence in both gene expression and histology in situ and effectively integrating data from multi-sample remain challenging. Here, we propose ResST, a graph self-supervised residual learning model based on graph neural network and Margin Disparity Discrepancy (MDD) theory. ResST aggregates gene expression, biological effects, spatial location, and morphological information to capture nonlinear relationships between a cell and surrounding cells for spatial domain identification. Also, ResST integrates multiple ST datasets and aligns latent embeddings based on MDD theory for correcting batch effects. Results show that ResST identifies continuous spatial domains at a finer scale in ten ST datasets acquired with different technologies. Moreover, ResST efficiently integrated data from multiple tissue sections vertically or horizontally while correcting batch effects. Overall, ResST demonstrates exceptional performance in analyzing ST datasets.

# Requirements

