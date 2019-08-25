This directory shows the results of the trainings.

The crnn_results and unet_results each contatin the Jupyter notebooks of all the trainings of each network for each three under-sampling. Note that the data used was coming from different sources, in different formats and shapes. In the first part of the jupyter notebooks I first read the files and then keep reshaping and reformating the data with constant checking by visualing the data. Also a lot of the code used here are research code, so for a more detailed and commented code check the Models directory.

After the training the testing data was reconstructed and they were saved for running the quantitative assessments.

The file all_quant shows my result of all the quantitative assessments using the saved images from the reconstructions.
Note that the file is 100 Mb. So, if you wanna have a look at it, download it.
