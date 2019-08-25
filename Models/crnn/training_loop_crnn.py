import os
import datetime
import torch

'''
The training loop used for training the CRNN.
The parameters and names depend on a person's preference.
'''

# Gets the GPU as device, So ,to(device) fo rall transfers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_vec = []
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    """
    Trainign loop used to train the CRNN. 
    You have to define the number of epochs.
    You have to define the model, optimizer, loss function and the 
    train loader which are defined before the training.
    """
    print('saving epoch {%d}'%0)
    #Saves the first initial model
    checkpoint = {'model': CRNN_MRI(), 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, 'checkpoint_crnn_13und_epoch__%d.pth'%0)
    for epoch in range(1, n_epochs + 1):
        i = 0
        loss_train = 0
        for imgs, labels, k, m in train_loader:

            # Make sure the inputs are all floats
            # Otherwise use .float()

            imgs = imgs.to(device)
            labels = labels.to(device)
            k = k.to(device)
            m = m.to(device)
            #Inputs to GPU


            outputs = model(imgs, k, m)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            #Hard clipping of gradients to overcome
            #the vanishing gradient problem
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)

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
