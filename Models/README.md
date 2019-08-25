# The scripts for the model
This directory contains all the clean codes for the models commented and explained. To see them in action, check the Results directory

# Contents:
unet: scripts for the unet training

crnn: scritps for the crnn training

The other files are codes that I used to quantitatively assess the results (image_quality_metrics.py) and loading the checkpoints 
(model_checkpoint_loader.py)



## U-net architecture

![](unet.png)


## CRNN architecture
Figure taken from Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction
Chen Qin, Jo Schlemper, Jose Caballero, Anthony Price, Joseph V. Hajnal, Daniel Rueckert

Architecture of the CRNN-MRI network for MRI reconstruction. Structure of the network when unfolded over the iterations. Structure of the BCRNN- t-i when unrolled over the time steps. Green arrows denote the feed-forward convolutions denoted by Wl . Blue arrows and red arrows denote the recurrent convolutions over iterations and the time-steps respectively. Note that in imple- mentation the weights are independent across layers, but here we used a single notation to denote weights of convolutions at different layers for the sake of simplicity.

![](crnn.png)
