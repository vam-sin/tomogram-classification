# Tomogram Classification

A deep learning solution to classify CryoET tomograms into various protein classes.

# Requirements

```python3
pip3 install mrcfile
pip3 install keras
pip3 install tensorflow
```

# Data

Ten different proteins were simulated in a cryo-electron tomogram. These tomograms were generated at three different Signal to Noise Ratios (SNR); infinity, 0.003, and 0.001. The data from the SHREC'19 track was used in this study. 

# Models

Four different models, the SqueezeNet, DenseNet 201, U-Net and the MobileNet were deisgned to classify each subtomogram into their respective proteins. The code to run the models is given below.

```python3
python3 densenet_train.py
python3 squeezenet_train.py
python3 mobilenetv2_train.py
python3 unet_train.py
```

The best results belonged to the SqueezeNet, they've been outlined below:

| SNR      | 0.003 | 0.005 | Infinity | 
| ----------- | ----------- | ----------- | ----------- | 
| SqueezeNet   | 72.20%         | 76.00%         | 79.20% |

# Paper

If you found anything in this repository useful, please consider citing the following paper.

```
@INPROCEEDINGS {9313185,
author = {S. Liu and Y. Ma and X. Ban and X. Zeng and V. Nallapareddy and A. Chaudhari and M. Xu},
booktitle = {2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
title = {Efficient Cryo-Electron Tomogram Simulation of Macromolecular Crowding with Application to SARS-CoV-2},
year = {2020},
volume = {},
issn = {},
pages = {80-87},
keywords = {proteins;viruses (medical);covid-19;tomography;three-dimensional displays;data models;pandemics},
doi = {10.1109/BIBM49941.2020.9313185},
url = {https://doi.ieeecomputersociety.org/10.1109/BIBM49941.2020.9313185},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {dec}
}
```
