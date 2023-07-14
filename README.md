# STGIN_main

![本地路径](fig/Figure_1.png)

# Overview
Spatial transcriptomics (ST) technologies are powerful experimental methods for measuring gene expression while preserving the relevant spatial context. However, existing methods for spatial clustering of ST ignore the cellular biological effects,  which are gene expression and spatial information of cells or spots containing complex global interactions within the tissue. Also, a key obstacle to integrate ST data from multiple sections vertically or horizontally is the presence of batch effects. Current methods used to data integration clustered multiple samples into the same layered structure, generating spots with mixed information, resulting in lacking layer-specific distinctions. To address these issues, we developed a nonlinear deep neural network model incorporated with Margin Disparity Discrepancy measurement, STGIN, which takes into account the influences of gene expression, spatial location, complex global interactions, and tissue morphology information on the features of individual cells within tissue sections, resulting in more natural and realistic spatial clustering performance. Moreover, STGIN efficiently conducted vertical or horizontal integration from multiple tissue sections while correcting batch effects and preserving layer-specific gene expression variations. Also, STGIN can make detailed spatial domain divisions for ST data from different platforms. In conclusion, STGIN has an exceptional ability in analyzing ST datasets.

# Requirements

