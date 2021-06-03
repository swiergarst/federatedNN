### imports
import torch 
from .model import model




### master task (will not be used in the current setup, as the researcher file will do this)
def master_task():
    pass




### RPC task
### This will contain the main training loop, including the forward and backward passes
def RPC_train(data, architecture, weights):
    ### create net from given architeture
    net = model(architecture)

    ### initialize the weights from input


    ### forward pass


    ### backward pass

    ### return the new weights