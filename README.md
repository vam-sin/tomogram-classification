# Tomogram Classification

A deep learning solution to classify CryoET tomograms into various protein classes.

# Requirements

```python3
pip3 install mrcfile
pip3 install keras
pip3 install tensorflow
```

# Data

Ten different proteins were simulated in a cryo-electron tomogram. These tomograms were generated at three different signal-to-noise ratios; infinity, 0.003, and 0.001. The data from the SHREC'19 track was used in this study. 

# Models

Two different models, the SqueezeNet, DenseNet 201, U-Net and the MobileNet were deisgned to classify each subtomogram into their respective proteins. The code to run the models is given below.

```python3
python3 densenet_train.py
python3 squeezenet_train.py
python3 mobilenetv2_train.py
python3 unet_train.py
```
