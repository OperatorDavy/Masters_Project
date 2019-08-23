# Improving deep-learning methods for inverse problems
#### David Wilson
This repository contains my codes and jupyter notebooks showing the results of my models.
This code is part of MSc Scientific computing final project.
I implemented two deep-learning models in pytorch.

### Contents


### Dependencies:
See `REQUIREMENTS.txt` for a list of dependencies.

### Data:
In the data creation part, I've provided the Matlab scripts for generating the data I used in this project. I have mentioned if the code is not mine and provided by Great Ormond street hospital. Also, I did not put the data since they are medical and there are many complicated issues if I shared them. Also, the gridding file is not provided, this is because the file was too large for github and it seems to be proprietary code, so I chose not to actually put it there.

### Instructions:
I did all the training and and assessment fo the results in the Google's cloud. I used jupyter notebooks. Each time, I load and prepare the results manually. This is because there was different data coming from different sources each one having some different shapes. This was the best approach to avoid making mistakes that can cause hours of GPU time or give bad results because the data is not in the correct shape.

### References
-"Real‐time cardiovascular MR with spatio‐temporal artifact suppression using deep learning–proof of concept in congenital heart disease".
Andreas Hauptmann  Simon Arridge  Felix Lucka  Vivek Muthurangu  Jennifer A. Steeden

-"Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction".
Chen Qin, Jo Schlemper, Jose Caballero, Anthony Price, Joseph V. Hajnal, Daniel Rueckert
