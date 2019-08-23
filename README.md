# Improving deep-learning methods for inverse problems
#### David Wilson
This repository contains my codes and jupyter notebooks showing the results of my models.
This code is part of MSc Scientific computing final project.
I implemented two deep-learning models in PyTorch: U-net and convolutional recurrent neural network (CRNN).
I used jupyter notebooks in the Google cloud for training of all the models, so there was no command line interface used.

### Contents
Models: Contatins the scripts for our models, the training loop scripts. 

Results: Contatins the jupyter notebooks that trained and tested the networks.

Data: Contatins the Matlab scripts for genereating the synthetic training data. Note that I did not put the actual data-set used, since they are medical and there are many complicated issues if I shared them. Also, the gridding file which has all the code for the gridding is not provided, this is because the file was too large for github and it seems to be proprietary code, so I chose not to actually put it there.



### Dependencies:
See `REQUIREMENTS.txt` for a list of dependencies.


### Instructions:
I did all the training and and assessment fo the results in the Google's cloud. I used jupyter notebooks. Each time, I load and prepare the results manually. This is because there was different data coming from different sources each one having some different shapes. This was the best approach to avoid making mistakes that can cause hours of GPU time or give bad results because the data is not in the correct shape.

### References
-"Real‐time cardiovascular MR with spatio‐temporal artifact suppression using deep learning–proof of concept in congenital heart disease".
Andreas Hauptmann  Simon Arridge  Felix Lucka  Vivek Muthurangu  Jennifer A. Steeden

-"Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction".
Chen Qin, Jo Schlemper, Jose Caballero, Anthony Price, Joseph V. Hajnal, Daniel Rueckert
