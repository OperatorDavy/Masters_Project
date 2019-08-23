import os
import datetime
import torch



def load_checkpoint(model, optimizer, filename='ckp_crnn_13und_epoch_80.pth'):
    """
    This function is used to re-store the states of the model and
    the optimisation model at a the specified epoch
    """
    # Note: Input model & optimizer should be pre-defined.
    # This routine only updates their states.
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer

# To use the function just use the following code
# Note that they should be transferred to GPU (do model.to(device))
# where device was defined before

model, optimizer = load_checkpoint(model, optimizer) 
