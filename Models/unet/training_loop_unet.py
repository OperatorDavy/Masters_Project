import os
import datetime
import torch

'''
The training loop used for training the U-net.
The parameters and names depend on a person's preference.
'''

# Gets the GPU as device, So ,to(device) fo rall transfers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


loss_vec = []
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    """
    The Training loop which is used for training the models in PyTorch.
    Need to define the number of epochs, the optimizer and model initially 
    defined. the loss function defined and train_loader made which has the dataset
    for PyTorch
    """
    print('saving epoch {%d}'%0)
    #Saves the first initial model
    checkpoint = {'model': CRNN_MRI(), 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, 'checkpoint_crnn_13und_epoch__%d.pth'%0)
    
    for epoch in range(1, n_epochs + 1):
        i = 0
        loss_train = 0
        for imgs, labels in train_loader:

            # Make sure the inputs are all floats
            # Otherwise use .float()

            imgs = imgs.to(device)
            labels = labels.to(device)

            #Inputs to GPU


            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            
            # The next stops are how Pytorch uses the 
            # Backprop and the optimisation to minimise the loss
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_train += loss.item()

            # Remove this if you want
            # I used to chek how the model is doing every 100 iterations
            if i%100 == 0:
                print("Epoch: {}, Iteration: {}, Loss: {}, time: {}".format(epoch, i+1, loss_train, datetime.datetime.now()))
            i +=1
        loss_vec.append(loss_train)
        print(i)
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, float(loss_train)))
        

        # I save all the weights after the Epoch
        # Change the variables if you want for the name of the saved files
        print('saving epoch {%d}'%epoch)
        checkpoint = {'model': CRNN_MRI(), 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
        torch.save(checkpoint, 'ckp_crnn_13und_epoch_%d.pth'%epoch)

# Just run the loop and the model gets trained.
